# Recipe Data NER using Conditional Random Fields

This repository contains a Named Entity Recognition (NER) pipeline built to extract structured entities—quantities, units, and ingredients—from raw, unstructured recipe text. 

## Business Objective
Digital culinary platforms manage millions of recipes authored in natural language. Without structured parsing, platforms cannot offer ingredient-based search filters, accurate allergen checks, or dynamic portion scaling. This project automates that information extraction process.

## Technical Approach
Instead of evaluating tokens in isolation, the pipeline uses a Conditional Random Field (CRF) to model the conditional probability distribution over the entire label sequence. This captures the strict grammatical transitions in culinary data (e.g., Quantity -> Unit -> Ingredient).

**Tools Used:** Python, `sklearn-crfsuite`, `spaCy`, `pandas`

### Feature Engineering
The feature extractor builds a dictionary for every token incorporating three levels of linguistic information:
* **Core Lexical & Grammatical:** Syntactic roles, base lemmas, and punctuation markers extracted via spaCy (`en_core_web_sm`).
* **Quantity & Unit Detectors:** Custom regular expressions for compound fractions (e.g., 1-1/2), simple fractions, and decimals, mapped against domain-specific unit keyword sets.
* **Contextual & Boundary Indicators:** Surrounding window context (previous/next tokens) and sentence boundaries (BOS/EOS) to guide sequence transitions.

To address a severe 6.5-to-1 class imbalance between ingredients and units, the pipeline applies inverse frequency class weighting.

## Performance Metrics
The model was evaluated on a strict, disjoint 30% validation split (84 unseen recipes, 2,876 tokens).

* **Overall Token Accuracy:** 98.05%
* **Macro F1-Score:** 0.9684

**Entity-Level Breakdown:**
* **Ingredient:** Precision 98.35% | Recall 99.15% | F1 0.9875
* **Quantity:** Precision 98.78% | Recall 98.78% | F1 0.9878
* **Unit:** Precision 95.31% | Recall 90.78% | F1 0.9299

## Error Analysis
The model produced exactly 56 misclassifications out of 2,876 validation tokens (1.95% error rate). Over 82% of these errors occurred at the boundary between units and ingredients. 

Primary failure modes:
1. **Polysemous Terms:** Words like "cloves" function as both a count unit ("4 cloves Garlic") and a distinct ingredient ("2 Cloves Laung"). Brief contexts cause the model to default to the majority ingredient class.
2. **Annotation Inconsistencies:** Procedural adverbs and verbs (e.g., "finely", "cut") were occasionally tagged as units in the ground truth dataset, conflicting with their true syntactic function.
