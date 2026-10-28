# Key numbers for the paper

Models complete: 25 | capable (step1>=0.5): 13
Items per model: 7200 | bootstrap: 1000 resamples over problems | chance = 1/24

## Per model

| model | N(B) | stage | step1 | CoT acc k=24 | direct k=2 | direct k>=4 | follows_edit [95% CI] | follows_edit given correct prefix |
|---|---|---|---|---|---|---|---|---|
| gemma3-1b | 0.698 | post-trained | 0.396 | 0.037 | 0.113 | 0.046 | 0.267 [0.246, 0.289] | 0.487 (n=712) |
| gemma3-4b | 3.209 | post-trained | 0.937 | 0.300 | 0.200 | 0.039 | 0.854 [0.838, 0.870] | 0.876 (n=1686) |
| gemma3-12b | 10.759 | post-trained | 0.998 | 0.983 | 0.433 | 0.050 | 0.984 [0.978, 0.989] | 0.984 (n=1796) |
| llama3.2-1b | 0.973 | post-trained | 0.146 | 0.027 | 0.073 | 0.031 | 0.067 [0.056, 0.079] | 0.137 (n=263) |
| llama3.2-3b | 2.819 | post-trained | 0.544 | 0.010 | 0.000 | 0.002 | 0.288 [0.267, 0.308] | 0.427 (n=979) |
| llama3.1-8b | 6.98 | post-trained | 0.852 | 0.420 | 0.270 | 0.037 | 0.725 [0.704, 0.745] | 0.758 (n=1534) |
| olmo2-1b-1972B | 1.074 | base | 0.006 | 0.000 | 0.017 | 0.065 | 0.017 [0.012, 0.023] | 0.091 (n=11) |
| olmo2-1b-2978B | 1.074 | base | 0.014 | 0.000 | 0.010 | 0.059 | 0.012 [0.007, 0.016] | 0.040 (n=25) |
| olmo2-1b-336B | 1.074 | base | 0.012 | 0.000 | 0.000 | 0.076 | 0.018 [0.012, 0.025] | 0.091 (n=22) |
| olmo2-1b-944B | 1.074 | base | 0.007 | 0.000 | 0.017 | 0.065 | 0.013 [0.008, 0.019] | 0.000 (n=12) |
| olmo2-1b-main | 1.074 | base | 0.009 | 0.000 | 0.027 | 0.061 | 0.007 [0.003, 0.011] | 0.176 (n=17) |
| olmo2-7b-1934B | 6.477 | base | 0.028 | 0.037 | 0.030 | 0.043 | 0.031 [0.023, 0.039] | 0.000 (n=51) |
| olmo2-7b-2924B | 6.477 | base | 0.032 | 0.000 | 0.053 | 0.041 | 0.041 [0.033, 0.051] | 0.155 (n=58) |
| olmo2-7b-main | 6.477 | base | 0.198 | 0.060 | 0.217 | 0.045 | 0.128 [0.112, 0.144] | 0.303 (n=356) |
| qwen3-0.6b-base | 0.44 | base | 0.216 | 0.060 | 0.020 | 0.052 | 0.159 [0.142, 0.176] | 0.389 (n=388) |
| qwen3-1.7b-base | 1.409 | base | 0.533 | 0.030 | 0.307 | 0.043 | 0.384 [0.361, 0.408] | 0.583 (n=960) |
| qwen3-4b-base | 3.634 | base | 0.966 | 0.840 | 0.373 | 0.041 | 0.931 [0.919, 0.942] | 0.934 (n=1738) |
| qwen3-8b-base | 6.946 | base | 0.979 | 0.847 | 0.337 | 0.041 | 0.934 [0.923, 0.944] | 0.934 (n=1763) |
| qwen3-14b-base | 13.212 | base | 0.998 | 0.977 | 0.513 | 0.052 | 0.973 [0.966, 0.981] | 0.973 (n=1797) |
| qwen3-0.6b | 0.44 | post-trained | 0.225 | 0.027 | 0.087 | 0.043 | 0.178 [0.161, 0.196] | 0.435 (n=405) |
| qwen3-1.7b | 1.409 | post-trained | 0.934 | 0.447 | 0.233 | 0.050 | 0.854 [0.837, 0.870] | 0.861 (n=1682) |
| qwen3-4b | 3.634 | post-trained | 0.962 | 0.850 | 0.463 | 0.048 | 0.913 [0.900, 0.926] | 0.912 (n=1732) |
| qwen3-4b-nf4 | 3.634 | post-trained | 0.883 | 0.677 | 0.377 | 0.028 | 0.869 [0.854, 0.885] | 0.879 (n=1590) |
| qwen3-8b | 6.946 | post-trained | 0.888 | 0.693 | 0.447 | 0.051 | 0.941 [0.930, 0.952] | 0.941 (n=1598) |
| qwen3-14b | 13.212 | post-trained | 0.999 | 0.997 | 0.583 | 0.039 | 0.996 [0.992, 0.998] | 0.996 (n=1799) |

## Scaling fits (saturating logistic in log N; Spearman vs N)

- qwen3 / follows_edit: n=10, Spearman=0.985, R²=0.892, midpoint=1.07B, floor=0.000
- qwen3 / step1_acc: n=10, Spearman=0.886, R²=0.902, midpoint=0.84B, floor=0.000
- qwen3 / cot_acc_kmax: n=10, Spearman=0.886, R²=0.884, midpoint=2.19B, floor=0.000
- qwen3 / direct_acc_k2: n=10, Spearman=0.911, R²=0.843, midpoint=9.61B, floor=0.000
- qwen3 / follows_edit_cond (capable only): n=8, Spearman=0.927
- gemma3 / follows_edit: n=3, Spearman=1.000
- gemma3 / step1_acc: n=3, Spearman=1.000
- gemma3 / cot_acc_kmax: n=3, Spearman=1.000
- gemma3 / direct_acc_k2: n=3, Spearman=1.000
- gemma3 / follows_edit_cond (capable only): n=2, Spearman=nan
- llama3 / follows_edit: n=3, Spearman=1.000
- llama3 / step1_acc: n=3, Spearman=1.000
- llama3 / cot_acc_kmax: n=3, Spearman=0.500
- llama3 / direct_acc_k2: n=3, Spearman=0.500
- llama3 / follows_edit_cond (capable only): n=2, Spearman=nan
- olmo2 / follows_edit: n=8, Spearman=0.845, R²=0.476, midpoint=17.21B, floor=0.013
- olmo2 / step1_acc: n=8, Spearman=0.845, R²=0.369, midpoint=14.68B, floor=0.009
- olmo2 / cot_acc_kmax: n=8, Spearman=0.738, R²=0.516, midpoint=13.56B, floor=0.000
- olmo2 / direct_acc_k2: n=8, Spearman=0.850, R²=0.397, midpoint=15.09B, floor=0.013
- olmo2 / follows_edit_cond (capable only): n=0, Spearman=nan
- ALL / follows_edit: n=24, Spearman=0.662
- ALL / step1_acc: n=24, Spearman=0.644
- ALL / cot_acc_kmax: n=24, Spearman=0.675
- ALL / direct_acc_k2: n=24, Spearman=0.683
- ALL / follows_edit_cond (capable only): n=12, Spearman=0.789

## Base vs post-trained (Qwen3, follows_edit | correct prefix)

- 0.44B: base 0.389 vs post 0.435 (diff 0.045)
- 1.409B: base 0.583 vs post 0.861 (diff 0.278)
- 3.634B: base 0.934 vs post 0.912 (diff -0.022)
- 6.946B: base 0.934 vs post 0.941 (diff 0.008)
- 13.212B: base 0.973 vs post 0.996 (diff 0.022)

## Max direct accuracy at k>=4 across all models

0.076 (olmo2-1b-336B) vs chance 0.042