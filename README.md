# Recovering unreported PE/VC funds from business-scope text

Code sample from a research assistant project. The notebook `pevc_recovery.ipynb` identifies private equity and venture capital (PE/VC) funds in Chinese business registration data from the text of their registered business scope, and recovers funds that do not appear in the self-reported registry of the Asset Management Association of China (AMAC).

## The problem

AMAC registration is self-reported, so it misses many funds that act as PE/VC investors. There is no ground-truth label: whether a firm counts as PE/VC is a human judgment, so a standard supervised classifier cannot simply be trained on a labelled data set. The algorithm instead learns what PE/VC business scopes look like from the text itself.

## The pipeline

1. **Remove boilerplate.** Rule-based prefix cleaning, then a fine-tuned Chinese RoBERTa token classifier (`hfl/chinese-roberta-wwm-ext`) that labels and removes the disclaimer suffixes, with a regex fallback layer and row-level audit flags.
2. **Onion peeling.** Embed each business-scope statement (`bge-m3`) and peel off outliers layer by layer until the variance of the remaining core converges. What survives is the set of core PE/VC statements. The notebook compares the result with Local Outlier Factor.
3. **Dictionary map.** Split each business scope into items and map every item to the core statements by embedding similarity (threshold 0.85), keeping the position of each item.
4. **Encoder.** Build firm-level vectors from the mapped items to identify PE/VC funds, including those that never registered with AMAC.

## Running it

The notebook was written for Google Colab with an L4 GPU (sections 2 and the BERT steps). Paths are set in the first code cell. The registration data are not included. Comments and notes are in English; the Chinese strings that remain in the code are matching keywords for the Chinese text and must not be translated.
