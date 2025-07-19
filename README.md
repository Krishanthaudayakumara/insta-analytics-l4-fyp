# Instagram User Behavior Analysis - Research Project

## Overview
This project analyzes user behavior and engagement on Instagram using data science and machine learning. It covers data processing, sentiment analysis, clustering, engagement prediction, visualization, and AI-driven recommendations.

## Workflow
1. **Data Processing**: Extract and merge Instagram post data with influencer metadata.
2. **Data Cleaning**: Remove duplicates, handle missing values, and standardize columns.
3. **Sentiment Analysis**: Analyze captions for sentiment using NLP.
4. **User Segmentation**: Cluster users based on engagement metrics.
5. **Engagement Prediction**: Predict comments/engagement using regression models.
6. **Visualization**: Generate plots for trends, distributions, and insights.
7. **Recommendations**: Suggest optimal post strategies based on data.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare your data in the `data/` folder (see project context).
3. Run the main pipeline:
   ```bash
   python main.py
   ```
4. Outputs (plots, logs, results) will be saved in `outputs/`, `visualizations/`, and `logs/`.

## How to Run the Web UI
You can launch an interactive dashboard for this project using Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This will open a web interface where you can upload data, run the analysis pipeline, view visualizations, and see recommendations.

## Project Structure
- `main.py`: Orchestrates the workflow and logs progress.
- `scripts/`: Data extraction, merging, and cleaning scripts.
- `analysis/`: Sentiment analysis, clustering, and prediction modules.
- `visualizations/`: Visualization scripts.
- `recommendations/`: Recommendation logic.
- `notebooks/`: Jupyter notebooks for exploration.
- `data/`: Input and processed data (not tracked in git).
- `outputs/`, `visualizations/`, `logs/`: Results and logs.

## Research Context
See `COPILOT_PROJECT_CONTEXT.md` for a detailed research background, module breakdown, and scientific context.

## Logging & Reproducibility
All major steps log to `logs/project.log` and print to console. Errors are handled gracefully for research reproducibility.

---

For more details, see the full project context and module breakdown in `COPILOT_PROJECT_CONTEXT.md`.
