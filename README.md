# BERT-based Text Analysis Application

Welcome to the **BERT-based Text Analysis Application** – a cutting-edge tool for performing robust text similarity analysis and factor analysis using state-of-the-art language models. This project combines the efficiency of BERT-based sentence embeddings with the analytical power of Python’s data science libraries, offering a comprehensive and interactive experience for researchers, data scientists, and analysts alike.

---

## Features

- **Multiple Similarity Scoring Methods:**  
  Choose from aggregated scoring, item-by-item scoring, or single-construct analysis for deep text evaluation.

- **Dynamic Text Embeddings:**  
  Generate text embeddings using powerful SentenceTransformer models like `all-MiniLM-L6-v2`, `paraphrase-MiniLM-L3-v2`, and `all-distilroberta-v1`.

- **Interactive Data Exploration:**  
  Visualize your data through descriptive statistics, histograms, and interactive correlation heatmaps with Plotly.

- **Advanced Statistical Analysis:**  
  Perform outlier exclusion, normalization, and correlation analysis with significance testing (p-values).

- **Factor Analysis Capabilities:**  
  Leverage EFA or PCA to uncover underlying factors in your data, complete with optional scree plots for deeper insights.

- **User-Friendly Interface:**  
  Built with Streamlit, the application offers an intuitive, web-based interface that facilitates easy configuration, analysis, and data download.

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.7+
- [pip](https://pip.pypa.io/en/stable/installation/)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/bert-text-analysis-app.git
   cd bert-text-analysis-app
   ```

2. **Install dependencies:**

   Use the provided `requirements.txt` to install all necessary libraries:

   ```bash
   pip install -r requirements.txt
   ```

   **Note:**  
   Ensure that the versions of `torch` and `torchvision` are compatible. In this project, `torch==2.2.0` and `torchvision==0.17.0` are used to maintain dependency consistency.

3. **Run the Application:**

   Launch the Streamlit app:

   ```bash
   streamlit run main.py
   ```

---

## Usage

1. **Upload Your Data:**  
   Use the sidebar to upload your text data (CSV or Excel format) and select the column that contains your text.

2. **Select Scoring Method:**  
   Choose between:
   - **Aggregated Items (Excel Upload)**
   - **Item-by-item (Excel Upload)**
   - **Single Construct (Interactive Input)**
   
   For Excel-based methods, upload your validated scales file. For interactive input, add your constructs directly through the interface.

3. **Generate Embeddings:**  
   Click the "Generate Text Embeddings" button to compute sentence embeddings using your selected model.

4. **Compute Similarity Scores:**  
   Once embeddings are generated, click "Compute Similarity Scores" to calculate similarity measures between your text data and your constructs or scale items.

5. **Data Analysis:**  
   - **Exclude Outliers:** Remove extreme values using Z-score filtering.
   - **Normalize Data:** Apply Min-Max normalization.
   - **Descriptive & Correlation Analysis:** View statistical summaries and visualize correlation matrices.
   - **Download Dataset:** Save the enhanced dataset as a CSV file.

---

## Contributing

Contributions are welcome! Whether you find a bug, have an idea for a new feature, or want to improve documentation, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For inquiries, feedback, or collaboration opportunities, please reach out:

- **LinkedIn:** [Gabriele Di Cicco](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)

---

Harness the power of BERT-based text analysis to unlock new insights from your data. Happy analyzing!
