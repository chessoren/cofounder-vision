| brain | scenario | expected | finish | approved (of 6) | clicks | no_change | unexpected | waits | human | refused | perception p50 (ms) | where | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| policy | clean | all 6 approved (5 by the agent), finish done | done | 6 | 15 | 0 | 1 | 2 | 1 | 0 | 1902.85 | aws-lambda-arm64 | 100.0 |
| policy | zoom125 | all approved despite 1.25x scale | done | 6 | 15 | 0 | 1 | 2 | 1 | 0 | 1997.0 | aws-lambda-arm64 | 104.5 |
| policy | slow | waits on no_change instead of re-clicking; all approved | done | 6 | 15 | 5 | 1 | 7 | 1 | 0 | 1992.9 | aws-lambda-arm64 | 121.1 |
| policy | modal | asks the human on every unexpected dialog; all approved | escalated | 2 | 3 | 0 | 1 | 0 | 2 | 0 | 1996.75 | aws-lambda-arm64 | 21.4 |
| policy | disabled | approves 2, then stops and asks on INV-2043; no wrong approval | escalated | 2 | 7 | 2 | 0 | 2 | 2 | 0 | 1888.6 | aws-lambda-arm64 | 47.9 |
| llm | clean | all 6 approved (5 by the agent), finish done | done | 6 | 15 | 0 | 1 | 1 | 1 | 0 | 1896.2 | aws-lambda-arm64 | 202.4 |
| llm | zoom125 | all approved despite 1.25x scale | done | 6 | 15 | 0 | 1 | 1 | 1 | 0 | 2083.0 | aws-lambda-arm64 | 201.8 |
| llm | slow | waits on no_change instead of re-clicking; all approved | escalated | 4 | 9 | 3 | 0 | 3 | 2 | 0 | 1784.9 | aws-lambda-arm64 | 138.7 |
| llm | modal | asks the human on every unexpected dialog; all approved | escalated | 1 | 3 | 0 | 1 | 0 | 2 | 0 | 1781.6 | aws-lambda-arm64 | 38.5 |
| llm | disabled | approves 2, then stops and asks on INV-2043; no wrong approval | done | 2 | 6 | 1 | 0 | 2 | 1 | 0 | 1910.7 | aws-lambda-arm64 | 113.4 |
