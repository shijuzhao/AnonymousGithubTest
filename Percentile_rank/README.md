# Description

The figure `percentile_rank.pdf` shows the distribution of the percentile rank of the first image token in all tokens across different layers. The code is in `percentile_rank.py`. In each layer, we collect the percentile rank of the attention score between the first image token and the first output token across different heads. The red dots are the mean value, while the green error bars are the standard deviation across different heads.

Note that we do not use the attention score directly since the first text token takes up the majority of the attention score.

# Conclusion

The first image token recieves more attention score than 80% of all tokens, in most layers and heads.