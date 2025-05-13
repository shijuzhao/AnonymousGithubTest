# Supplementary Material for InfoBlend

This folder contains the dataset, code, and results of Figure 4 in the paper. The results of two MLLMs are in their respective folders.

# Code Structure

```
InfoBlend
├── llava-v1.6
│   ├── attn_distribution
│   ├── cumulative_attn
├── InternVL-2.5
│   ├── attn_distribution
│   └── cumulative_attn
├── attn_distribution.py
├── configs.py
├── cumulative_attn.py
└── example.py
```

# Code
- `configs.py` includes the experimental settings.
- `example.py` includes the instance to be investigated.
- `attn_distribution.py` investigates the distribution of attention scores.
- `cumulative_attn.py` investigates the importance of initial image tokens.

# Results
1. Less than 5% of tokens receive more than $10^{-3}$ attention score. Since $10^{-3}$ is small, this means that less than 5% of tokens affect the output.
2. The first 500 tokens account for approximately 80% of attention scores. The initial tokens tend to attract a disproportionate share of attention.