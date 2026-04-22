# 📰 Text Mining on CNN/Daily Mail: Topic Modeling & Extractive Summarization

> **University Project — Academic Year 2025/2026**  
> Stephen Adu Poku Yeboah (852121) · Emmanuel Ampah (868745)

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-News%20Topic%20Analyzer-blue)](https://huggingface.co/spaces/sapyeboa/News_topic_analyzer)
[![Dataset](https://img.shields.io/badge/Dataset-CNN%2FDaily%20Mail-orange)](https://huggingface.co/datasets/abisee/cnn_dailymail)
[![Python](https://img.shields.io/badge/Python-3.x-green)](https://www.python.org/)

---

## 🔍 Overview

This project explores two core text mining tasks on the **CNN/Daily Mail news corpus** (92,579 articles):

1. **Topic Modeling** — Unsupervised discovery of latent themes using LDA, LSA, and BERTopic
2. **Extractive Summarization** — Selecting the most salient sentences using graph-based (TextRank + TF-IDF) and embedding-based (Word2Vec + TF-IDF) approaches

A live interactive demo is deployed on Hugging Face Spaces, allowing you to input any news article and see how it maps to the discovered topics.

---

## 🚀 Live Demo

👉 **[News Topic Analyzer on Hugging Face](https://huggingface.co/spaces/sapyeboa/News_topic_analyzer)**

Paste any news article and the app will classify it against the LDA, LSA, and BERTopic models trained on the CNN/Daily Mail corpus, showing topic distributions and top keywords.

---

## 📂 Repository Structure

```
├── topic-modelling.ipynb          # Topic modeling: LDA, LSA, BERTopic (hyperparameter tuning + evaluation)
├── text-summarisation-graph.ipynb # Extractive summarization: TextRank + Word2Vec
└── README.md
```

---

## 📊 Dataset

| Statistic | Value |
|---|---|
| Total Documents | 92,579 |
| Avg. Tokens per Document | 312.7 |
| Vocabulary (filtered) | 10,000 terms |
| Avg. Highlights per Article | 3–4 |

The corpus covers politics, international affairs, sports, entertainment, and human interest stories. Each article includes human-written highlights used as reference summaries for ROUGE evaluation.

---

## 🧠 Methods

### Topic Modeling

Three approaches were implemented and compared:

| Model | # Topics | Coherence (C_V) | Coherence (U_Mass) |
|---|---|---|---|
| **LDA** | 5 | **0.4271** | -2.1219 |
| LSA | 5 | 0.3760 | -2.2958 |
| BERTopic | 155 | N/A | N/A |

**LDA** achieved the best interpretable coherence. The 5 discovered topics include:

- 🌍 International Politics & Conflicts
- 🚔 Crime & Law Enforcement
- 🇺🇸 US Politics & Sports
- 🏙️ Business & Urban Development
- 🎬 Entertainment & Lifestyle

**BERTopic** discovered 155 fine-grained, event-level topics — useful for granular news classification, though with ~30.6% outlier documents.

#### Key preprocessing steps
- Custom stop word list (314 words)
- Bigram detection via Gensim Phrases (optimal: 86,286 unique phrases)
- Dictionary filtering: `no_below=10`, `no_above=0.5`, `keep_n=10,000` → 67.66% vocabulary coverage

---

### Extractive Summarization

Two methods were implemented and evaluated with ROUGE scores against human-written highlights:

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| **TextRank + TF-IDF** | **0.3624** | **0.0896** | **0.1536** |
| Word2Vec + TF-IDF | 0.1933 | 0.0792 | 0.1295 |

**TextRank (Graph-Based):**
Sentences are represented as TF-IDF vectors, connected in a similarity graph, and scored via PageRank. Top-ranked sentences form the summary.

**Word2Vec + TF-IDF (Centroid-Based):**
Word2Vec embeddings (Skip-gram, `vector_size=150`, `window=8`, `epochs=10`) are weighted by TF-IDF to form sentence vectors. Sentences are ranked by cosine similarity to the document centroid; the top 40% are selected.

---

## ⚙️ Setup & Usage

```bash
# Clone the repository
git clone https://github.com/<stephyeboah>/<text-mining-cnn-dailymail>.git
cd <repo-name>

# Install dependencies
pip install gensim bertopic sentence-transformers rouge-score nltk scikit-learn

# Run notebooks
jupyter notebook topic-modelling.ipynb
jupyter notebook text-summarisation-graph.ipynb
```

---

## 📈 Key Findings

- **LDA** with a small number of topics (5) best captures dominant editorial themes in a news corpus
- **BERTopic** excels at fine-grained, event-level discovery but requires careful outlier handling
- **TextRank** outperforms the centroid-based approach across all ROUGE metrics, benefiting from the global sentence-graph structure
- TF-IDF weighting improves embedding-based sentence salience estimation

---

## 📚 References

1. Hermann et al. (2015). *Teaching Machines to Read and Comprehend.* NeurIPS.
2. Blei, Ng & Jordan (2003). *Latent Dirichlet Allocation.* JMLR.
3. Grootendorst (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794.
4. Mihalcea & Tarau (2004). *TextRank: Bringing order into text.* EMNLP.
5. Mikolov et al. (2013). *Efficient estimation of word representations in vector space.* arXiv:1301.3781.
