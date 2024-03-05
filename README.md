# DDIET_Tarea5

-----

# Document vector representation task 📄

## Overview 👁
This repository contains the results and code for our exploration into document vector representations, utilizing various models to analyze text in English and Spanish. Our study focuses on the implementation of additive, average, TF, and TF-IDF models to create meaningful vector representations of documents.

## Models used 🧠
- English: [Google's word2vec](https://code.google.com/archive/p/word2vec/)
- Spanish: [Spanish Billion Words Corpus and Embeddings (SBWCE)](https://crscardellino.github.io/SBWCE/)

Each model was chosen for its ability to capture the linguistic nuances of the respective language, producing vectors with 300 elements that reflect a broad spectrum of semantic information.

## Results ✏
The repository includes the following result files for both English and Spanish document collections:
- `additive_web_en.txt`
- `additive_web_es.txt`
- `average_web_en.txt`
- `average_web_es.txt`
- `tfidf_web_en.txt`
- `tfidf_web_es.txt`
- `tf_web_en.txt`
- `tf_web_es.txt`

Each file corresponds to a different model and language, containing vector representations for five documents. The additive and average models provide sentence-by-sentence representations, while the TF and TF-IDF models offer per-word representations.


## Conclusion ✨
This project underlines the importance of choosing the appropriate vector representation model based on the linguistic and semantic requirements of the analysis. The chosen models and methodologies demonstrate robustness in capturing the essence of document semantics across languages.
