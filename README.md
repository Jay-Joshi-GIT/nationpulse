# NationPulse

**An explainable model versus a black-box LLM for scoring economies and planning growth.**

NationPulse runs two systems head to head on 52 countries:

- **System 1: XGBoost + SHAP.** A gradient-boosted tree model scores each country's economy from seven indicators, and SHAP explains every score.
- **System 2: an LLM.** Mistral 7B, run locally with Ollama, gets the same seven indicators and gives its own score and explanation.

Neither system judges the other. Both are checked against the real World Bank score, the pace of change countries have actually achieved, standard economic direction, and computing cost. The output is a scoreboard of which system does better at what (`scoreboard.png`).

Course project for ES 678, IIT Gandhinagar, by Jay Prafulkumar Joshi (26210062).

## Main results (2023 data, 52 countries)

| | XGBoost + SHAP | LLM (Mistral 7B) |
|---|---|---|
| Error on countries it never trained on (points, 0–100 scale) | **9.1** | 17.8 |
| Rank agreement with the real score (Spearman) | **0.87** | 0.63 |
| Worst income group, average error | **+8.4** | +33.6 |
| Score change when the question is reworded | **0** | 13.8 |
| Growth plans at a realistic pace | **100%** (by design) | 65% |
| Growth plans moving indicators the way economics recommends | 74% | **100%** |
| Cost for both jobs, 52 countries (estimated) | **4.1 s, 0.017 Wh** | 163 min, 40.8 Wh |

For comparison, a plain linear regression is 13.3 points off and guessing the average is 21.8.

## How to run

You need Python 3.12 and [Ollama](https://ollama.com) with the Mistral model (`ollama pull mistral`).

```bash
python -m venv NPenv
source NPenv/bin/activate
pip install -r requirements.txt
```

Then run the steps in this order. The LLM steps take over an hour each on a laptop CPU.

| Step | Command | Makes |
|---|---|---|
| 1 | `python data_pipeline.py` | `economic_data.csv` (World Bank + IMF data) |
| 2 | `python xgboost_scorer.py` | `xgb_model.json`, `xgb_scores.csv` |
| 3 | `python shap_explainer.py` | `shap_scores.csv`, `shap_summary_plot.png` |
| 4 | `python llm_scorer.py` | `llm_scores.csv` (asks Mistral) |
| 5 | `python comparison.py` | `comparison*.csv`, charts |
| 6 | `python save_results.py` | `conformal_bounds.csv`, copies everything to `results/` |
| 7 | `python strategy_planner.py all` | `strategy_plans.csv`, `strategy_ratings.csv` (asks Mistral) |
| 8 | `python environmental_cost.py` | `environmental_cost.csv`, chart |
| 9 | `python scoreboard.py` | `scoreboard.csv`, `scoreboard.png` |
| | `python dashboard.py` | Web dashboard at http://127.0.0.1:7860 |

The results of every step are already in this repository, so you can open the dashboard or the scoreboard without re-running the LLM.

## Data and credits

The data comes from the World Bank World Development Indicators (CC BY 4.0) and the IMF World Economic Outlook. These and their original sources (IMF, ILO, UN, UNESCO, OECD) keep their own terms; some data may not be reused commercially without permission. See [CREDITS.md](CREDITS.md) for every data source, model, library and method, with licences and citations.

No data provider endorses this project. The code was written with the help of Claude Code (Anthropic).
