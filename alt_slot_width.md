# Default slot width: 60 s vs 10 s (tau = 0)

500 queries per regime and window; same physical windows; seed 17.

## Witness statistics (60-min window)

| Regime | Slot | Contacts | Reachable | Median delay [min] | Witness median/max | Size reduction | Certified |
|---|---|---|---|---|---|---|---|
| Full ISLs | 60 s | 46,334 | 500/500 | 0.0 | 6/10 | 83.3% | 500/500 |
| Full ISLs | 10 s | 287,280 | 500/500 | 0.0 | 6/10 | 83.3% | 500/500 |
| Intra-plane | 60 s | 34,090 | 500/500 | 0.0 | 8/21 | 77.3% | 500/500 |
| Intra-plane | 10 s | 211,816 | 500/500 | 0.0 | 7/21 | 77.5% | 500/500 |
| No ISLs | 60 s | 10,330 | 310/500 | 26.0 | 4/12 | 98.8% | 500/500 |
| No ISLs | 10 s | 69,256 | 403/500 | 26.0 | 6/16 | 99.8% | 500/500 |

## Error of time-agnostic semantics (reachable / static error / snapshot error, %)

| Regime | Slot | 15 min | 30 min | 60 min |
|---|---|---|---|---|
| Full ISLs | 60 s | 100.0 / 0.0 / 10.0 | 100.0 / 0.0 / 6.0 | 100.0 / 0.0 / 6.0 |
| Full ISLs | 10 s | 100.0 / 0.0 / 2.8 | 100.0 / 0.0 / 1.8 | 100.0 / 0.0 / 1.4 |
| Intra-plane | 60 s | 100.0 / 0.0 / 14.8 | 100.0 / 0.0 / 16.8 | 100.0 / 0.0 / 15.0 |
| Intra-plane | 10 s | 100.0 / 0.0 / 5.2 | 100.0 / 0.0 / 6.4 | 100.0 / 0.0 / 5.8 |
| No ISLs | 60 s | 18.2 / 62.2 / 15.6 | 35.0 / 65.0 / 33.6 | 63.6 / 36.4 / 62.8 |
| No ISLs | 10 s | 24.6 / 61.0 / 21.6 | 42.4 / 57.6 / 40.8 | 81.6 / 18.4 / 79.4 |
