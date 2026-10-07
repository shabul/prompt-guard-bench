# AnyJev calibrated threshold evaluation

This calibration experiment used three disjoint, stratified sets:

| Set | Rows | Purpose |
|---|---:|---|
| Calibration | 210 | Select threshold only |
| Test | 70 | Primary held-out evaluation |
| OOB | 71 | Second untouched audit set |

Dataset SHA-256: 1ed7f83b96fe6c0592b764ef1094df7abec97989df76d67074d1fe5023733115  
Model: google/gemma-3-270m-it  
AnyJev: L0, canonical ordering  
Backend: Transformers HFBackend on Apple M5 MPS  
Numeric type: float32

## Calibration

The threshold was selected from calibration probabilities only, with a target precision of at least 90%.

- Selected injection threshold: 0.63756
- Calibration precision: 100.00%
- Calibration recall: 1.09%
- Calibration F1: 2.15%

This threshold marked only 1 of 92 calibration injections as INJECTION.

## Held-out results

| Metric | Test | OOB |
|---|---:|---:|
| Accuracy | 55.71% | 56.34% |
| Precision | 0.00% | 0.00% |
| Recall | 0.00% | 0.00% |
| F1 | 0.00% | 0.00% |
| Injection TP / FP / FN / TN | 0 / 0 / 31 / 39 | 0 / 0 / 31 / 40 |
| Mean latency | 61.8 ms | 57.3 ms |

## Conclusion

The 90% precision requirement can be met on calibration only by choosing a threshold that is too high to generalize. Both untouched sets demonstrate zero recall at that threshold. This is a valid calibration result, not a successful classifier configuration.

The next sensible experiment is to choose a threshold by maximizing calibration F1 or by enforcing a less aggressive precision target, then report that operating point on both held-out sets. The data split and model probabilities remain separate and auditable in anyjev_calibrated_results.json locally.
