| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA | deepseek-v4-flash | 80% | 1.34M | 57k | 1.39M | 11.47 |
| Lifelong AgentBench | GA | deepseek-v4-flash | 90% | 649k | 42k | 690k | 26.07 |

RealFin-benchmark was not added as a full Table 2 row in this run. A 1-task smoke run failed before producing the required output file because the local environment lacks finance data packages and direct Eastmoney requests were disconnected (`RemoteDisconnected` / `ProxyError`). The failed smoke used 53,431 input tokens and 2,557 output tokens.
