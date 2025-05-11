# CASE-Bench
This repository contains the Context-Aware SafEty Benchmark (CASE-Bench). 
![CASE-Bench](teaser.png)
---
Aligning large language models (LLMs) with human values is essential for their safe deployment and widespread adoption. Current LLM safety benchmarks often focus solely on the refusal of individual problematic queries, which overlooks the importance of the context where the query occurs and may cause undesired refusal of queries under safe contexts that diminish user experience. Addressing this gap, we introduce CASE-Bench, a Context-Aware SafEty Benchmark that integrates context into safety assessments of LLMs. CASE-Bench assigns distinct, formally described contexts to categorized queries based on Contextual Integrity theory. Additionally, in contrast to previous studies which mainly rely on majority voting from just a few annotators, we recruited a sufficient number of annotators necessary to ensure the detection of statistically significant differences among the experimental conditions based on power analysis. Our extensive analysis using CASE-Bench on various open-source and commercial LLMs reveals a substantial and significant influence of context on human judgments ($p<$0.0001 from a z-test), underscoring the necessity of context in safety evaluations. We also identify notable mismatches between human judgments and LLM responses, particularly in commercial models within safe contexts.

<div style='display:flex; gap: 0.25rem; '>
<a href='[https://arxiv.org/abs/2502.11775](https://arxiv.org/pdf/2501.14940)'><img src='https://img.shields.io/badge/arXiv-PDF-red'></a>
</div>

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

Note: This dataset is for review only as it includes queries from Sorry-bench and access to these queries must comply with the researchers' agreement and require granted access on HuggingFace. Accordingly, the anonymized link provided below is strictly for review purposes only. Upon publication, we will grant access to our dataset exclusively to users who have obtained permission to access the Sorry-bench dataset, thereby ensuring adherence to the original dataset's ethical guidelines.

| :exclamation:  The "Child-related Crimes" category is consistently labelled as unsafe, regardless of context, following Sorry-Bench. For a detailed discussion, see the Impact Statement in the paper.   |
|-----------------------------------------|


## Results
Please find the outputs from models in `exp/`

## Useful Scripts
Useful scripts to reproduce the experiments:
```
infer_api.py: Inference code for API-based models
infer_model.py: Inference code for opensource models
compute_scores.py: compute accuracy, PCC and BCE, etc.
```

## Reference
```
@inproceedings{
  sun2025casebench,
  title={{CASE-Bench}: {C}ontext-{A}ware {S}af{E}ty {B}enchmark for {L}arge {L}anguage {M}odels},
  author={Guangzhi Sun, Xiao Zhan, Shutong Feng, Philip C Woodland, Jose Such},
  booktitle={ICML},
  year={2025}
}
```
