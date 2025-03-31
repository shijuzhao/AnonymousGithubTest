# File structure

The main result is shown in `SEEME.png`, and its components (legend and two subfigures) are the `PDF` files.

# Results

1. As the request rate increases, the TTFT of serving systems rises significantly. This phenomenon can be attributed to the fact that, in high-demand scenarios, requests are required to queue, thereby delaying the processing of individual requests.
2. As the request rate increases, the throughput of serving systems first rises linearly, then falls, and finally stabilizes around a fixed value. The throughput rises linearly because the low request rate cannot saturate the system. When the request rate exceeds the capability of the system, the throughput falls and converges to a fixed value.
3. InfoBlend consistently outperforms CacheBlend and prefix caching in terms of TTFT and throughput.