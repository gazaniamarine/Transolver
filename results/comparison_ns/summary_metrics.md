# Quantitative Performance Comparison Summary

Resolution: 128x128

| Metric | FNO Baseline | Transolver | Comparison |
| :--- | :---: | :---: | :---: |
| **Test Relative L2 Error** | 0.14414 ± 0.01740 | 0.39500 ± 0.03989 | FNO lower by 2.74x |
| **Parameter Count** | 2,803,009 | 1,420,577 | Transolver has 2.0x fewer |
| **Avg. Inference Latency (per sample)** | 2.04 ms | 7.80 ms | FNO faster by 3.82x |
| **Mesh-Independence** | No (Rigid spatial grid representation) | Yes (Arbitrary query coordinates) | Transolver learns continuous spatial operators |
