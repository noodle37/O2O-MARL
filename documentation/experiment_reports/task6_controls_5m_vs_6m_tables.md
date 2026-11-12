# 5m_vs_6m simple controls: tables

| arm | eval points | AUC 0-0.5M | AUC 0-1M | full AUC (0-3M) | tail win (2.5-3M) | last win | 640-ep win [95% CI] | tail reg | tail eta |
|---|---|---|---|---|---|---|---|---|---|
| Fixed Joint-Ranking (FJR) | 298 | 0.011 | 0.130 | 0.415 | 0.618 | 0.531 | 0.617 [0.579,0.654] | 0.01605 | 0.100 |
| Per-agent Distillation (PAD) | 299 | 0.011 | 0.124 | 0.406 | 0.622 | 0.688 | 0.637 [0.600,0.674] | 0.00620 | 0.100 |
| Time-Decayed Ranking (TDR) | 299 | 0.002 | 0.024 | 0.474 | 0.801 | 0.906 | 0.834 [0.804,0.861] | 0.00000 | 0.000 |
| Per-agent marginal support | 298 | 0.011 | 0.116 | 0.419 | 0.624 | 0.562 | 0.645 [0.607,0.681] | 0.01784 | 0.100 |
| State-only support | 298 | 0.028 | 0.152 | 0.421 | 0.634 | 0.719 | 0.658 [0.620,0.694] | 0.01506 | 0.100 |
| No support | 298 | 0.026 | 0.163 | 0.428 | 0.633 | 0.688 | 0.620 [0.582,0.657] | 0.01645 | 0.100 |

| arm | 640-ep Δ vs FJR | Δ 95% half-width | tail Δ vs FJR | ranking scope | support | S_off |
|---|---|---|---|---|---|---|
| Per-agent Distillation (PAD) | 0.020 | 0.053 | 0.004 | agent | joint | 3283 |
| Time-Decayed Ranking (TDR) | 0.217 | 0.047 | 0.183 | joint | joint | 3283 |
| Per-agent marginal support | 0.028 | 0.053 | 0.006 | joint | marginal | 3283 |
| State-only support | 0.041 | 0.053 | 0.016 | joint | state | 3283 |
| No support | 0.003 | 0.053 | 0.015 | joint | none | 3283 |
