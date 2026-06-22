# psyBert

A Streamlit application that scores open-ended text responses against Likert-type scale items using semantic similarity.

Given a set of texts (e.g., survey open-ends) and a set of scale items per construct, the app encodes both with SentenceTransformers (SBERT) and computes the mean cosine similarity of each text to each construct's items — optionally reversing similarity for reverse-coded items. No fine-tuning or classification involved: it is a lookup between the embedding spaces of responses and scale anchors.

[![CI](https://github.com/gdc0000/psyBert/actions/workflows/ci.yml/badge.svg)](https://github.com/gdc0000/psyBert/actions/workflows/ci.yml)

**Live app:** [psy-bert.streamlit.app](https://psy-bert.streamlit.app/)

---

## What it does

1. **Embed** — encodes each response into a fixed-size vector using a SentenceTransformer model.
2. **Score** — for each construct, embeds its scale items and computes cosine similarity between every response and every item. Returns the mean similarity across items (aggregated mode) or each item individually (item-by-item mode).
3. **Reverse** — items flagged as reversed are scored as `1 - similarity`, consistent with Likert-scale convention.
4. **Analyze** — descriptive statistics and a Pearson correlation matrix with significance stars across the computed scores.
5. **Export** — download the similarity table as CSV.

---

## Getting Started

### Prerequisites

- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/installation/)

### Installation

```bash
git clone https://github.com/gdc0000/psyBert.git
cd psyBert
pip install -r requirements.txt
streamlit run main.py
```

Alternatively, install in editable mode:

```bash
pip install -e ".[dev]"
```

---

## Usage

1. Upload text data (CSV or Excel) and select the text column in the sidebar.
2. Choose a scoring method — upload a validated scales Excel file or define constructs interactively.
3. Generate embeddings in the **Embeddings** tab.
4. Compute similarity scores in the **Similarity** tab.
5. View descriptive statistics and the annotated correlation matrix in the **Analysis** tab.
6. Download results from the **Download** tab.

---

## Project Structure

```
app/
├── __init__.py
├── analysis.py    # Correlation and significance
├── config.py      # Streamlit page config
├── data.py        # File loading
├── ml.py          # Embedding and similarity computation
├── services.py    # Orchestration layer (UI-independent)
├── settings.py    # Constants
├── state.py       # Session state management
└── ui/            # Streamlit components
    ├── __init__.py
    ├── footer.py
    ├── sidebar.py
    └── tabs.py
tests/             # Pytest test suite
main.py            # Entrypoint
```

---

## License

MIT
