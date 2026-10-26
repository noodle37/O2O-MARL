# FHJP vs OVMSE vs UOT (auto-generated)

Generated: 2026-10-26 14:16:38

## Summary

- **5m_vs_6m**: FHJP endpoint win 0.689 ± 0.055 over 2 completed seed(s) (return 16.91 ± 0.57).
  - FHJP s1 − UOT-A s1: Δwin +0.002 [-0.048, 0.050] (not resolved), Δreturn 0.01 [-0.44, 0.45].
  - FHJP s1 − OVMSE-C s1: Δwin -0.008 [-0.055, 0.039] (not resolved), Δreturn -0.12 [-0.55, 0.31].
  - FHJP s1 − OVMSE-D s1: Δwin +0.289 [0.247, 0.336] (favours FHJP), Δreturn 2.82 [2.38, 3.29].
  - FHJP s1 − OVMSE-paper s1: Δwin +0.061 [0.016, 0.106] (favours FHJP), Δreturn 0.59 [0.16, 1.02].
  - FHJP s1 − vanilla-B s1: Δwin +0.180 [0.127, 0.236] (favours FHJP), Δreturn 1.95 [1.42, 2.51].
  - FHJP s1 − mix-E s1: Δwin +0.186 [0.133, 0.236] (favours FHJP), Δreturn 1.91 [1.39, 2.38].
  - Proposer exit: seed1: version 2 at 1000k steps, seed2: version 2 at 1000k steps, seed3: version 4 at 2000k steps (run incomplete); after exit the run is plain online QMIX TD.
- **2s3z**: FHJP endpoint win 0.990 ± 0.003 over 3 completed seed(s) (return 19.94 ± 0.02).
  - FHJP s1 − OVMSE-paper s1: Δwin +0.019 [0.005, 0.031] (favours FHJP), Δreturn 0.14 [0.05, 0.23].
  - Proposer exit: seed1: version 1 at 200k steps, seed2: version 1 at 200k steps, seed3: version 2 at 400k steps; after exit the run is plain online QMIX TD.
- The proposer's `offline_select_rate` stays at ~0.50 inside conflict decisions on every run, matching the fixed p=0.5 randomisation, and `conflict_rate` falls across versions (~0.53→0.43 on 5m_vs_6m), i.e. the online policy converges toward the offline joint actions.

## Protocol

| item | value |
|---|---|
| primary endpoint | 640-episode evaluation of the final online checkpoint (chunk 32, Wilson 95% CI) |
| 5m_vs_6m online budget | 3,000,000 env steps |
| 2s3z online budget | 1,000,000 env steps |
| shared offline init (both maps) | `checkpoints/ovmse_batch_20261014/<map>_offline/ovmse_paper_seed1_best.pt` |
| FHJP recipe | plain online QMIX TD + 0.5/0.5 joint proposer at decision conflicts; no CQL, no OVM target, no offline replay |
| FHJP proposer | p=0.5, version_len=500k (5m_vs_6m) / 200k (2s3z), M=32 episodes per arm, M_min=16, alpha=0.05, eval steps counted in the budget |
| eval seed | equal to the run's online seed; arms sharing an eval seed are contrasted with a paired chunk bootstrap |

## Arms

| arm | algorithm | recipe | run directory |
|---|---|---|---|
| FHJP (5m_vs_6m) | FHJP | joint proposer p=0.5 + plain online QMIX TD (no CQL, no OVM, no offline replay) | `checkpoints/fhjp_5m_vs_6m/seed1` |
| UOT-A (5m_vs_6m) | UOT | matched arm A: UOT eta=0.1, OVM off, mixing 0 | `checkpoints/matched_protocol_20261020/A_seed1` |
| OVMSE-C (5m_vs_6m) | OVMSE | matched arm C: OVM on, offline replay mixing 0.1 | `checkpoints/matched_protocol_20261020/C_seed1` |
| OVMSE-D (5m_vs_6m) | OVMSE | matched arm D: OVM on, offline replay mixing 0 | `checkpoints/matched_protocol_20261020/D_seed1` |
| OVMSE-paper (5m_vs_6m) | OVMSE | paper arm (configs/5m_vs_6m.yaml), same offline init as every other arm here | `checkpoints/ovmse_batch_20261014/5m_vs_6m_online` |
| vanilla-B (5m_vs_6m) | vanilla online QMIX | matched arm B: no UOT, no OVM, no mixing (FHJP's recipe without the proposer) | `checkpoints/matched_protocol_20261020/B_seed1` |
| mix-E (5m_vs_6m) | ablation | matched arm E: OVM off, offline replay mixing 0.1 | `checkpoints/matched_protocol_20261020/E_seed1` |
| FHJP (2s3z) | FHJP | joint proposer p=0.5 + plain online QMIX TD (no CQL, no OVM, no offline replay) | `checkpoints/fhjp_2s3z/seed1` |
| OVMSE-paper (2s3z) | OVMSE | paper arm (configs/2s3z.yaml), same offline init as every other arm here | `checkpoints/ovmse_batch_20261014/2s3z_online` |

## Primary endpoint by map

### 5m_vs_6m

| arm | seed | win rate | 95% CI | mean return | ckpt t_env | endpoint |
|---|---:|---:|---|---:|---:|---|
| FHJP | 1 | 0.728 | [0.692, 0.761] | 17.32 | 3000069 | yes |
| FHJP | 2 | 0.650 | [0.612, 0.686] | 16.50 | 3000200 | yes |
| FHJP | 3 | pending | — | — | 2770524 | no (stopped manually at ~2.76M/3.0M steps; no final checkpoint, no 640-episode endpoint) |
| UOT-A | 1 | 0.727 | [0.691, 0.760] | 17.30 | 3000136 | yes |
| OVMSE-C | 1 | 0.736 | [0.700, 0.769] | 17.44 | 3000126 | yes |
| OVMSE-D | 1 | 0.439 | [0.401, 0.478] | 14.50 | 3000022 | yes |
| OVMSE-paper | 1 | 0.667 | [0.630, 0.703] | 16.73 | 3000050 | yes |
| vanilla-B | 1 | 0.548 | [0.510, 0.587] | 15.36 | 3000216 | yes |
| mix-E | 1 | 0.542 | [0.503, 0.580] | 15.41 | 3000118 | yes |

FHJP cross-seed (2 completed seed(s)): win 0.689 ± 0.055 (range 0.650–0.728), return 16.91 ± 0.57

### 2s3z

| arm | seed | win rate | 95% CI | mean return | ckpt t_env | endpoint |
|---|---:|---:|---|---:|---:|---|
| FHJP | 1 | 0.991 | [0.980, 0.996] | 19.96 | 1000051 | yes |
| FHJP | 2 | 0.992 | [0.982, 0.997] | 19.95 | 1000073 | yes |
| FHJP | 3 | 0.986 | [0.973, 0.993] | 19.92 | 1000024 | yes |
| OVMSE-paper | 1 | 0.972 | [0.956, 0.982] | 19.82 | 1000014 | yes |

FHJP cross-seed (3 completed seed(s)): win 0.990 ± 0.003 (range 0.986–0.992), return 19.94 ± 0.02

## Contrasts vs FHJP

Positive Δ means FHJP is better. Paired = same eval seed (chunk-level paired bootstrap); unpaired = different eval seed (Newcombe for win rate, chunk bootstrap for return).

| map | contrast | Δwin (95% CI) | Δreturn (95% CI) |
|---|---|---:|---:|
| 5m_vs_6m | FHJP s1 − UOT-A s1 | +0.002 [-0.048, 0.050] (paired) | +0.01 [-0.44, 0.45] (paired) |
| 5m_vs_6m | FHJP s1 − OVMSE-C s1 | -0.008 [-0.055, 0.039] (paired) | -0.12 [-0.55, 0.31] (paired) |
| 5m_vs_6m | FHJP s1 − OVMSE-D s1 | +0.289 [0.247, 0.336] (paired) | +2.82 [2.38, 3.29] (paired) |
| 5m_vs_6m | FHJP s1 − OVMSE-paper s1 | +0.061 [0.016, 0.106] (paired) | +0.59 [0.16, 1.02] (paired) |
| 5m_vs_6m | FHJP s1 − vanilla-B s1 | +0.180 [0.127, 0.236] (paired) | +1.95 [1.42, 2.51] (paired) |
| 5m_vs_6m | FHJP s1 − mix-E s1 | +0.186 [0.133, 0.236] (paired) | +1.91 [1.39, 2.38] (paired) |
| 5m_vs_6m | FHJP s2 − UOT-A s1 | -0.077 [-0.127, -0.026] (unpaired) | -0.80 [-1.20, -0.38] (unpaired) |
| 5m_vs_6m | FHJP s2 − OVMSE-C s1 | -0.086 [-0.136, -0.035] (unpaired) | -0.93 [-1.36, -0.51] (unpaired) |
| 5m_vs_6m | FHJP s2 − OVMSE-D s1 | +0.211 [0.157, 0.263] (unpaired) | +2.00 [1.53, 2.49] (unpaired) |
| 5m_vs_6m | FHJP s2 − OVMSE-paper s1 | -0.017 [-0.069, 0.035] (unpaired) | -0.22 [-0.67, 0.24] (unpaired) |
| 5m_vs_6m | FHJP s2 − vanilla-B s1 | +0.102 [0.048, 0.154] (unpaired) | +1.14 [0.57, 1.71] (unpaired) |
| 5m_vs_6m | FHJP s2 − mix-E s1 | +0.108 [0.054, 0.161] (unpaired) | +1.09 [0.64, 1.52] (unpaired) |
| 2s3z | FHJP s1 − OVMSE-paper s1 | +0.019 [0.005, 0.031] (paired) | +0.14 [0.05, 0.23] (paired) |
| 2s3z | FHJP s2 − OVMSE-paper s1 | +0.020 [0.006, 0.037] (unpaired) | +0.13 [0.03, 0.23] (unpaired) |
| 2s3z | FHJP s3 − OVMSE-paper s1 | +0.014 [-0.002, 0.031] (unpaired) | +0.10 [0.01, 0.20] (unpaired) |

## FHJP proposer diagnostics (version boundaries)

tau_v = paired mean of (offline-arm return − online-arm return) over 32 shared-seed episodes per arm; exit when the one-sided 95% upper bound <= 0 and M >= 16. Exit is irreversible, and after exit the run is plain online QMIX TD.

| map/seed | version | t_env | tau_v | tau_v_ub | offline arm | online arm | conflict rate | offline select rate | eval env steps | exited |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5m_vs_6m/seed1 | 1 | 500178 | 3.174 | 3.970 | 9.36 | 6.19 | 0.530 | 0.501 | 1537 | no |
| 5m_vs_6m/seed1 | 2 | 1000278 | -2.122 | -0.988 | 9.78 | 11.90 | 0.451 | 0.499 | 1692 | yes |
| 5m_vs_6m/seed2 | 1 | 500060 | 0.037 | 1.213 | 9.36 | 9.33 | 0.526 | 0.501 | 1652 | no |
| 5m_vs_6m/seed2 | 2 | 1000177 | -2.456 | -0.994 | 9.78 | 12.23 | 0.433 | 0.500 | 1704 | yes |
| 5m_vs_6m/seed3 | 1 | 500175 | 1.436 | 2.300 | 9.36 | 7.93 | 0.561 | 0.500 | 1633 | no |
| 5m_vs_6m/seed3 | 2 | 1000311 | 1.395 | 2.541 | 9.78 | 8.38 | 0.424 | 0.500 | 1619 | no |
| 5m_vs_6m/seed3 | 3 | 1500509 | -0.174 | 1.203 | 9.59 | 9.76 | 0.440 | 0.500 | 1670 | no |
| 5m_vs_6m/seed3 | 4 | 2000714 | -2.387 | -1.125 | 8.93 | 11.32 | 0.469 | 0.500 | 1738 | yes |
| 2s3z/seed1 | 1 | 200089 | -1.349 | -0.621 | 13.10 | 14.45 | 0.682 | 0.497 | 3450 | yes |
| 2s3z/seed2 | 1 | 200046 | -1.622 | -1.039 | 13.10 | 14.72 | 0.707 | 0.501 | 3600 | yes |
| 2s3z/seed3 | 1 | 200024 | 11.667 | 12.352 | 13.10 | 1.43 | 0.699 | 0.501 | 4677 | no |
| 2s3z/seed3 | 2 | 400053 | -1.875 | -1.266 | 13.14 | 15.02 | 0.727 | 0.501 | 3595 | yes |

## Early vs late training-eval performance (32-episode evals, 250k bins)

| map | arm | seed | first 250k win / ret | last 250k win / ret |
|---|---|---:|---:|---:|
| 5m_vs_6m | FHJP | 1 | 0.023 / 6.6 | 0.674 / 16.8 |
| 5m_vs_6m | FHJP | 2 | 0.023 / 7.0 | 0.604 / 16.1 |
| 5m_vs_6m | FHJP | 3 | 0.029 / 7.4 | 0.594 / 16.0 |
| 5m_vs_6m | UOT-A | 1 | 0.019 / 6.6 | 0.673 / 16.8 |
| 5m_vs_6m | OVMSE-C | 1 | 0.040 / 9.2 | 0.682 / 16.9 |
| 5m_vs_6m | OVMSE-D | 1 | 0.028 / 8.5 | 0.448 / 14.5 |
| 5m_vs_6m | OVMSE-paper | 1 | 0.069 / 9.5 | 0.724 / 17.4 |
| 5m_vs_6m | vanilla-B | 1 | 0.017 / 5.7 | 0.522 / 15.1 |
| 5m_vs_6m | mix-E | 1 | 0.031 / 6.9 | 0.570 / 15.8 |
| 2s3z | FHJP | 1 | 0.668 / 17.2 | 0.979 / 19.9 |
| 2s3z | FHJP | 2 | 0.738 / 17.7 | 0.984 / 19.9 |
| 2s3z | FHJP | 3 | 0.454 / 14.0 | 0.973 / 19.8 |
| 2s3z | OVMSE-paper | 1 | 0.806 / 18.7 | 0.951 / 19.7 |

## Descriptive: full 32-episode training-eval curve, 250k bins

### 5m_vs_6m

| t_env bin | FHJP s1 win / ret | FHJP s2 win / ret | FHJP s3 win / ret | UOT-A s1 win / ret | OVMSE-C s1 win / ret | OVMSE-D s1 win / ret | OVMSE-paper s1 win / ret | vanilla-B s1 win / ret | mix-E s1 win / ret |
|---|---|---|---|---|---|---|---|---|---|
| 0k–250k | 0.023 / 6.6 | 0.023 / 7.0 | 0.029 / 7.4 | 0.019 / 6.6 | 0.040 / 9.2 | 0.028 / 8.5 | 0.069 / 9.5 | 0.017 / 5.7 | 0.031 / 6.9 |
| 250k–500k | 0.060 / 9.4 | 0.072 / 9.4 | 0.074 / 9.4 | 0.068 / 9.5 | 0.217 / 11.8 | 0.087 / 10.0 | 0.228 / 12.0 | 0.007 / 7.8 | 0.040 / 9.1 |
| 500k–750k | 0.233 / 11.8 | 0.244 / 11.9 | 0.147 / 10.7 | 0.122 / 10.4 | 0.357 / 13.5 | 0.174 / 11.4 | 0.314 / 13.0 | 0.024 / 8.6 | 0.172 / 11.2 |
| 750k–1000k | 0.352 / 13.3 | 0.346 / 13.3 | 0.263 / 12.4 | 0.246 / 12.0 | 0.448 / 14.4 | 0.246 / 12.1 | 0.406 / 13.8 | 0.090 / 10.0 | 0.296 / 12.7 |
| 1000k–1250k | 0.421 / 14.0 | 0.401 / 13.9 | 0.325 / 13.1 | 0.380 / 13.6 | 0.475 / 14.7 | 0.305 / 12.9 | 0.443 / 14.3 | 0.246 / 12.0 | 0.350 / 13.3 |
| 1250k–1500k | 0.480 / 14.7 | 0.425 / 14.2 | 0.411 / 14.1 | 0.385 / 13.7 | 0.568 / 15.7 | 0.306 / 13.0 | 0.514 / 15.2 | 0.357 / 13.3 | 0.390 / 13.8 |
| 1500k–1750k | 0.539 / 15.3 | 0.499 / 15.0 | 0.432 / 14.4 | 0.494 / 14.8 | 0.629 / 16.3 | 0.390 / 13.9 | 0.594 / 16.0 | 0.372 / 13.5 | 0.395 / 13.9 |
| 1750k–2000k | 0.572 / 15.6 | 0.514 / 15.1 | 0.450 / 14.6 | 0.568 / 15.5 | 0.621 / 16.2 | 0.421 / 14.2 | 0.648 / 16.6 | 0.454 / 14.3 | 0.445 / 14.4 |
| 2000k–2250k | 0.604 / 15.9 | 0.580 / 15.8 | 0.502 / 15.0 | 0.616 / 16.2 | 0.664 / 16.8 | 0.403 / 14.0 | 0.667 / 16.8 | 0.489 / 14.7 | 0.491 / 14.8 |
| 2250k–2500k | 0.651 / 16.5 | 0.604 / 16.0 | 0.604 / 16.2 | 0.606 / 16.1 | 0.642 / 16.5 | 0.400 / 14.1 | 0.675 / 16.9 | 0.524 / 15.1 | 0.515 / 15.1 |
| 2500k–2750k | 0.672 / 16.8 | 0.585 / 15.9 | 0.697 / 17.1 | 0.642 / 16.5 | 0.688 / 17.0 | 0.430 / 14.4 | 0.691 / 17.0 | 0.522 / 15.1 | 0.516 / 15.2 |
| 2750k–3000k | 0.674 / 16.8 | 0.604 / 16.1 | 0.594 / 16.0 | 0.673 / 16.8 | 0.682 / 16.9 | 0.448 / 14.5 | 0.724 / 17.4 | 0.522 / 15.1 | 0.570 / 15.8 |

### 2s3z

| t_env bin | FHJP s1 win / ret | FHJP s2 win / ret | FHJP s3 win / ret | OVMSE-paper s1 win / ret |
|---|---|---|---|---|
| 0k–250k | 0.668 / 17.2 | 0.738 / 17.7 | 0.454 / 14.0 | 0.806 / 18.7 |
| 250k–500k | 0.948 / 19.7 | 0.969 / 19.8 | 0.853 / 18.9 | 0.920 / 19.5 |
| 500k–750k | 0.986 / 19.9 | 0.981 / 19.9 | 0.975 / 19.8 | 0.953 / 19.7 |
| 750k–1000k | 0.979 / 19.9 | 0.984 / 19.9 | 0.973 / 19.8 | 0.951 / 19.7 |

## Reproduce

```bash
bash scripts/eval_fhjp.sh          # 640-episode endpoints for FHJP + OVMSE 2s3z
python scripts/analyze_fhjp_vs_arms.py --out logs/fhjp_vs_arms_summary.json \
    --md documentation/experiment_reports/fhjp_vs_ovmse_uot.md
```
