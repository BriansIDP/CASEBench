# CASE-Bench
This repository contains the Context-Aware Safety Evaluation Benchmark (CASE-Bench). 
![CASE-Bench](teaser.png)
---
Aligning large language models (LLMs) with human values is essential for their safe deployment and widespread adoption. Current LLM safety benchmarks often focus solely on the refusal of individual problematic queries, which overlooks the importance of context where the query occurs and may lead to over-refusal behaviours that diminish user experience. Addressing this gap, we introduce CASE-Bench, a Context-Aware Safety Evaluation Benchmark that integrates context into safety assessments of LLMs. CASE-Bench assigns distinct, formally described contexts to categorized queries based on Contextual Integrity theory. Additionally, in contrast to previous studies which mainly rely on the majority voting from only a few annotators, we recruited a sufficient number of annotators necessary to ensure the detection of significant differences among the experimental conditions based on power analysis. Our extensive analysis across various open-source and commercial LLMs reveals a substantial and significant influence of context on human judgments (p$<$0.0001 under a z-test), underscoring the necessity of context in safety evaluations. We also identify notable mismatches between human judgments and LLM responses, particularly in commercial models within safe contexts.


## Dataset
The data can be found [here](https://github.com/BriansIDP/CASEBench/blob/main/data/CASEBench_data.json). The field of each data sample is explained as follows:
```
"scores": Individual human judgment where 1 for safe and 2 for unsafe
"query": The original query
"context": The context, structured as a dictionary where the keys are CI parameters
"context_intended_to_be_safe": Whether the context was intended to be safe or unsafe when creating it
"safe_rate": Percentage of people choosing safe
"category": Category the query belongs to
```


## Results
Please find the outputs from models in `exp/`

## Useful Scripts
Useful scripts to reproduce the experiments:
```
infer_api.py: Inference code for API-based models
infer_model.py: Inference code for opensource models
compute_scores.py: compute accuracy, PCC and BCE, etc.
```
