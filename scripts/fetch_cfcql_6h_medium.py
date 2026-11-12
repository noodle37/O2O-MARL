#!/usr/bin/env python3
"""Fetch only the Medium split of the CFCQL 6h_vs_8z vault zip.

The full archive is ~348 MB (Medium + Medium-Replay + Mixed + Expert), but the
vault loader only opens one uid directory, so downloading everything wastes
hours on the rate-limited HF mirror. The zip central directory is fetched with
HTTP Range, then only the byte span covering ``6h_vs_8z.vlt/Medium/`` is
downloaded in resumable blocks and unpacked with CRC checks.

Output: vaults/cfcql/smac_v1/6h_vs_8z.vlt/Medium/{metadata.json,manifest.ocdbt,d/*}
"""

from __future__ import annotations

import concurrent.futures
import os
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

RESOLVE_URL = (
    "https://hf-mirror.com/datasets/InstaDeepAI/og-marl/resolve/main/"
    "prior_work/cfcql/smac_v1/6h_vs_8z.zip"
)
VAULT_BYTES = 364773563
ROOT = Path(__file__).resolve().parents[1]
VAULT_DIR = ROOT / "vaults/cfcql/smac_v1/6h_vs_8z.vlt"
WORK = ROOT / "vaults/tmp/6h_medium_range"
PREFIX = "6h_vs_8z.vlt/Medium/"
BLOCK = 1 << 18
WORKERS = 4


def log(msg: str) -> None:
    print(f"=== [{time.strftime('%F %T')}] {msg}", flush=True)


def resolve_url() -> str:
    for attempt in range(20):
        proc = subprocess.run(
            ["curl", "-sI", "--max-time", "60", RESOLVE_URL],
            capture_output=True,
            text=True,
        )
        for line in proc.stdout.splitlines():
            if line.lower().startswith("location:"):
                return line.split(":", 1)[1].strip()
        log(f"resolve attempt {attempt + 1} failed; sleeping")
        time.sleep(15)
    raise RuntimeError("could not resolve signed HF URL")


def curl_range(url: str, start: int, end: int, out: Path) -> None:
    expected = end - start + 1
    part = out.with_suffix(out.suffix + ".part")
    for attempt in range(120):
        if attempt and attempt % 5 == 0:
            # Signed URLs expire after ~1 h; refresh before retrying.
            url = resolve_url()
        # Always fetch the whole block from its start. The xet bridge can align
        # a resumed range to an internal chunk, which shifts the bytes; retrying
        # the 1 MiB block from scratch is cheaper than validating an offset.
        part.unlink(missing_ok=True)
        with open(part, "wb") as handle:
            proc = subprocess.run(
                [
                    "curl", "-s", "--fail", "--http1.1",
                    "--retry", "3", "--retry-delay", "5", "--retry-all-errors",
                    "--connect-timeout", "15", "--speed-limit", "1024", "--speed-time", "10",
                    "--max-time", "180",
                    "-r", f"{start}-{end}", url,
                ],
                stdout=handle,
                stderr=subprocess.DEVNULL,
            )
        if proc.returncode == 0 and part.exists() and part.stat().st_size == expected:
            os.replace(part, out)
            return
        time.sleep(min(5 + 5 * attempt, 45))
    raise RuntimeError(f"range {start}-{end} failed")


def curl_bytes(url: str, start: int, end: int) -> bytes:
    out = WORK / f"meta_{start}_{end}.bin"
    curl_range(url, start, end, out)
    return out.read_bytes()


def parse_central_directory(tail: bytes, tail_start: int):
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("zip EOCD not found")
    n_total, cd_size, cd_off = struct.unpack_from("<HII", tail, i + 10)
    pos = cd_off - tail_start
    entries = []
    for _ in range(n_total):
        if tail[pos:pos + 4] != b"PK\x01\x02":
            raise RuntimeError("bad central directory entry")
        (_sig, _vmade, _vneed, _flags, method, _mt, _md, crc, csize, usize,
         nlen, elen, clen, _disk, _iattr, _eattr, lho) = struct.unpack_from(
            "<IHHHHHHIIIHHHHHII", tail, pos
        )
        name = tail[pos + 46:pos + 46 + nlen].decode("utf-8")
        entries.append(
            dict(name=name, method=method, crc=crc, csize=csize, usize=usize,
                 nlen=nlen, elen=elen, lho=lho)
        )
        pos += 46 + nlen + elen + clen
    return entries


def main() -> int:
    if (VAULT_DIR / "Medium" / "metadata.json").is_file():
        log(f"already present: {VAULT_DIR}")
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    VAULT_DIR.parent.mkdir(parents=True, exist_ok=True)

    url = resolve_url()
    log("resolved signed URL")

    tail_len = min(262144, VAULT_BYTES)
    tail_start = VAULT_BYTES - tail_len
    tail = curl_bytes(url, tail_start, VAULT_BYTES - 1)
    entries = parse_central_directory(tail, tail_start)
    medium = [e for e in entries if e["name"].startswith(PREFIX)]
    if not medium:
        raise RuntimeError(f"no entries with prefix {PREFIX}")
    # metadata.json / manifest.ocdbt sit near the front of the archive while the
    # d/ chunks are a contiguous run further in. Take the span over d/ only and
    # fetch the two tiny files separately.
    chunks = [e for e in medium if "/d/" in e["name"] and e["csize"] > 0]
    span_start = min(e["lho"] for e in chunks)
    span_end = min(
        VAULT_BYTES - 1,
        max(e["lho"] + 30 + e["nlen"] + e["elen"] + e["csize"] for e in chunks) + 65536,
    )
    total = span_end - span_start + 1
    log(f"Medium entries={len(medium)} chunks={len(chunks)} "
        f"span={span_start}-{span_end} ({total / 2**20:.1f} MiB)")

    span_path = WORK / "medium_span.bin"
    if span_path.exists() and span_path.stat().st_size != total:
        span_path.unlink()
    if not span_path.exists():
        span_path.touch()
        os.truncate(span_path, total)

    n_blocks = (total + BLOCK - 1) // BLOCK
    done = 0
    t0 = time.time()
    lock = __import__("threading").Lock()

    def fetch_block(i: int) -> None:
        nonlocal done
        out = WORK / f"block_{i:05d}.bin"
        b0 = span_start + i * BLOCK
        b1 = min(span_end, b0 + BLOCK - 1)
        curl_range(url, b0, b1, out)
        with lock:
            with open(span_path, "r+b") as handle:
                handle.seek(i * BLOCK)
                handle.write(out.read_bytes())
            out.unlink()
            done += 1
            mb = (done * BLOCK) / 2**20
            rate = (done * BLOCK) / max(time.time() - t0, 1) / 1024
            log(f"blocks {done}/{n_blocks} ~{mb:.1f} MiB at {rate:.0f} KiB/s")

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(fetch_block, range(n_blocks)))

    span = span_path.read_bytes()
    tmp_vault = WORK / "6h_vs_8z.vlt"
    if tmp_vault.exists():
        import shutil
        shutil.rmtree(tmp_vault)
    for e in medium:
        dest = tmp_vault / Path(e["name"]).relative_to("6h_vs_8z.vlt")
        if e["name"].endswith("/"):
            dest.mkdir(parents=True, exist_ok=True)
            continue
        if span_start <= e["lho"] <= span_end:
            buf, base = span, span_start
        else:
            end = min(
                VAULT_BYTES - 1,
                e["lho"] + 30 + e["nlen"] + e["elen"] + e["csize"] + 512,
            )
            buf, base = curl_bytes(url, e["lho"], end), e["lho"]
        rel = e["lho"] - base
        if buf[rel:rel + 4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {e['name']}")
        nlen, elen = struct.unpack_from("<HH", buf, rel + 26)
        data_off = rel + 30 + nlen + elen
        comp = buf[data_off:data_off + e["csize"]]
        if e["method"] == 0:
            raw = comp
        elif e["method"] == 8:
            raw = zlib.decompress(comp, -15)
        else:
            raise RuntimeError(f"unsupported compression method {e['method']}")
        if zlib.crc32(raw) & 0xFFFFFFFF != e["crc"]:
            raise RuntimeError(f"CRC mismatch for {e['name']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)

    if VAULT_DIR.exists():
        import shutil
        shutil.rmtree(VAULT_DIR)
    os.replace(tmp_vault, VAULT_DIR)
    span_path.unlink()
    log(f"vault ready: {VAULT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"FATAL: {exc!r}")
        sys.exit(1)
