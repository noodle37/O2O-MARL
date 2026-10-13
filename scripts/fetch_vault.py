#!/usr/bin/env python3
"""Resumable byte-range vault downloader (HF mirror friendly).

The built-in data.vault_download streams the whole zip in one request, which
dies on multi-GB vaults over an unstable link. This fetches chunk-aligned byte
ranges in parallel, records holes via SEEK_HOLE so an interrupted run resumes
instead of restarting, then unzips into vaults/<source>/<env>/.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.vault_download import VAULT_INFO, check_directory_exists_and_not_empty
from utils.paths import VAULT_DIR

DEFAULT_MIRROR = "https://hf-mirror.com"
CHUNK = 4 * 1024 * 1024
_thread_local = threading.local()


def mirror_url(url: str, endpoint: str) -> str:
    if endpoint and "huggingface.co" in url:
        return url.replace("https://huggingface.co", endpoint).replace(
            "http://huggingface.co", endpoint
        )
    return url


def resolve(url: str) -> tuple[str, int]:
    """Follow redirects to the CDN object and report its true size."""
    response = requests.get(url, stream=True, timeout=(30, 60), allow_redirects=True)
    try:
        response.raise_for_status()
        final_url = response.url
        length = response.headers.get("Content-Length")
    finally:
        response.close()
    if not length:
        head = requests.head(final_url, timeout=(30, 60), allow_redirects=True)
        head.raise_for_status()
        final_url = head.url
        length = head.headers["Content-Length"]
    return final_url, int(length)


class SignedUrl:
    """Caches the redirected CDN URL, refreshed on a short TTL.

    Resolving per chunk costs an extra round trip for every one of thousands of
    ranges, but pinning one URL for the whole download is worse: the mirror
    302s to a single CDN edge, and enough parallel range requests to that one
    edge stall with the socket open and no bytes flowing. A short TTL keeps the
    round trips amortized while rotating workers across edges.
    """

    def __init__(self, page_url: str, ttl: float = 120.0):
        self._page_url = page_url
        self._ttl = ttl
        self._url = ""
        self._expiry = 0.0
        self._lock = threading.Lock()

    def get(self, force: bool = False) -> str:
        with self._lock:
            if force or not self._url or time.monotonic() > self._expiry:
                self._url, _ = resolve(self._page_url)
                self._expiry = time.monotonic() + self._ttl
            return self._url


def missing_ranges(path: Path, total: int, chunk: int) -> list[tuple[int, int]]:
    """Chunk-aligned ranges that are still holes in a sparse partial file."""
    if not path.is_file() or path.stat().st_size != total:
        return [(i, min(i + chunk - 1, total - 1)) for i in range(0, total, chunk)]
    fd = os.open(path, os.O_RDONLY)
    holes: list[tuple[int, int]] = []
    off = 0
    try:
        while off < total:
            try:
                data = os.lseek(fd, off, os.SEEK_DATA)
            except OSError:
                holes.append((off, total - 1))
                break
            if data > off:
                holes.append((off, data - 1))
            try:
                hole = os.lseek(fd, data, os.SEEK_HOLE)
            except OSError:
                break
            off = hole
    finally:
        os.close(fd)
    ranges: list[tuple[int, int]] = []
    for start, end in holes:
        pos = (start // chunk) * chunk
        while pos <= end:
            ranges.append((pos, min(pos + chunk - 1, total - 1)))
            pos += chunk
    return sorted(set(ranges))


def _session() -> requests.Session:
    """Per-thread session so pooled connections avoid a TLS handshake per chunk."""
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _thread_local.session = sess
    return sess


def fetch_one(
    signed: SignedUrl,
    path: Path,
    start: int,
    end: int,
    attempts: int,
    read_timeout: float = 45.0,
) -> None:
    expect = end - start + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            url = signed.get(force=attempt > 0)
            with _session().get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(15, read_timeout),
                stream=True,
            ) as response:
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {response.status_code}")
                data = response.content
            if len(data) != expect:
                raise RuntimeError(f"got {len(data)} bytes, expected {expect}")
            with open(path, "r+b") as handle:
                handle.seek(start)
                handle.write(data)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"retry {start}-{end} attempt {attempt + 1}: {exc}", flush=True)
            time.sleep(min(20, 2 * (attempt + 1)))
    raise RuntimeError(f"range {start}-{end} failed: {last_error}")


def download(
    page_url: str, path: Path, workers: int, chunk: int, read_timeout: float
) -> None:
    signed = SignedUrl(page_url)
    _, total = resolve(page_url)
    print(f"target={path} size={total} bytes ({total / 2**30:.2f} GiB)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size != total:
        with open(path, "wb") as handle:
            handle.truncate(total)
    ranges = missing_ranges(path, total, chunk)
    print(f"chunks_total={(total + chunk - 1) // chunk} chunks_missing={len(ranges)} "
          f"workers={workers} read_timeout={read_timeout}", flush=True)
    if not ranges:
        print("already complete", flush=True)
        return
    started = time.monotonic()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fetch_one, signed, path, start, end, 12, read_timeout)
            for start, end in ranges
        ]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 20 == 0 or done == len(ranges):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = done * chunk / elapsed / 2**20
                eta = (len(ranges) - done) * chunk / max(rate * 2**20, 1)
                print(f"progress {done}/{len(ranges)} ({100 * done / len(ranges):.1f}%) "
                      f"{rate:.2f} MiB/s eta {eta / 60:.1f} min", flush=True)
    size = path.stat().st_size
    if size != total:
        raise SystemExit(f"size mismatch {size} != {total}")
    with zipfile.ZipFile(path) as zf:
        if zf.testzip() is not None:
            # No hole remains to resume from, so a corrupt archive would loop
            # forever on restart; drop it and let the caller re-download.
            path.unlink()
            raise SystemExit("zip integrity check failed; removed partial archive")
    print(f"download complete {size}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="VAULT_INFO top key, e.g. og_marl")
    parser.add_argument("--env", required=True, help="e.g. mamujoco, smac_v1")
    parser.add_argument("--scenario", required=True, help="e.g. 2halfcheetah")
    parser.add_argument("--mirror", default=os.environ.get("HF_ENDPOINT", DEFAULT_MIRROR))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-mb", type=int, default=CHUNK // 2**20)
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=45.0,
        help="Per-chunk socket read timeout; a stalled CDN edge fails fast and re-resolves.",
    )
    parser.add_argument("--unzip", action="store_true")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    try:
        page_url = VAULT_INFO[args.source][args.env][args.scenario]["url"]
    except KeyError:
        raise SystemExit(
            f"{args.source}/{args.env}/{args.scenario} not in VAULT_INFO"
        ) from None
    page_url = mirror_url(page_url, args.mirror.rstrip("/"))
    print(f"url={page_url}", flush=True)

    zip_path = VAULT_DIR / "tmp" / f"{args.source}_{args.env}_{args.scenario}.zip"
    extraction_path = VAULT_DIR / args.source / args.env
    vault_path = extraction_path / f"{args.scenario}.vlt"

    if check_directory_exists_and_not_empty(str(vault_path)):
        print(f"vault already present: {vault_path}", flush=True)
        return

    download(page_url, zip_path, args.workers, args.chunk_mb * 2**20, args.read_timeout)

    if args.unzip:
        extraction_path.mkdir(parents=True, exist_ok=True)
        print(f"unzipping -> {extraction_path}", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extraction_path)
        if not check_directory_exists_and_not_empty(str(vault_path)):
            raise SystemExit(f"expected vault missing after unzip: {vault_path}")
        if not args.keep_zip:
            zip_path.unlink()
        print(f"vault ready: {vault_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
