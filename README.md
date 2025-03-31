# Anonymous Repository for InfoBlend

This repository contains additional experimental results.

# Code Structure

The folder `llava-v1.6` contains the dataset, code, and results of Figure 4 in the paper, while the folder `InternVL-2.5` contains that of another MLLM `InternVL 2.5`. The folder `Throuput_experiment` contains the evaluation results of InfoBlend under different request rates.

```
InfoBlend
├── llava-v1.6
│   ├── attn_distribution
│   ├── cumulative_attn
│   ├── attn_distribution.py
│   ├── configs.py
│   ├── cumulative_attn.py
│   └── example.py
├── InternVL-2.5
│   ├── attn_distribution
│   └── cumulative_attn
└── Throughput_experiment
    ├── SEEME.png
    ├── legend.pdf
    ├── Throughput.pdf
    └── TTFT.pdf
```