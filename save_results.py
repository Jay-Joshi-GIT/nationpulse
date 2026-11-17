# save_results.py  (Part 10 of the NationPulse guide)  —  Part B of the project
#
# 1. Gives XGBoost's scores a 95% range ("how sure am I?"), using two
#    machine learning methods:
#      - Quantile XGBoost: one model predicts the low edge, one the high edge
#      - Conformal prediction: pads the prediction by an error margin learned
#        from countries the model never saw (this gives the 95% guarantee)
#    The range is checked against the REAL score of countries the model never
#    saw. The LLM gives no range at all, so this is an ability only System 1 has.
#    (An earlier version used this range to mark LLM scores as TRUSTED or
#    FLAGGED. That made XGBoost the judge of the LLM, so it was removed.)
# 2. Copies every output file into the results/ folder.
#
# Input:  economic_data.csv (+ every earlier output file)
# Output: conformal_bounds.csv and the results/ folder
#
# Run:  python save_results.py

import os
import shutil
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from xgboost_scorer import feature_cols, latest_year, make_score


def compute_bounds(df):
    X = df[feature_cols]
    y = df['economic_score']

    # Split by COUNTRY: 70% of countries train the models, the other 30%
    # ("calibration" countries) are used only to measure how wrong they are.
    all_countries = sorted(df['country'].unique())
    train_countries, cal_countries = train_test_split(all_countries, test_size=0.3, random_state=42)
    is_train = df['country'].isin(train_countries)
    X_train, y_train = X[is_train], y[is_train]
    X_cal, y_cal = X[~is_train], y[~is_train]

    # ---- Method 1: Quantile XGBoost ----
    # quantile_alpha=0.05 learns the value that only 5% of countries fall below,
    # quantile_alpha=0.95 the value that only 5% fall above.
    lower_model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.05,
                               n_estimators=200, max_depth=3, random_state=42)
    upper_model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.95,
                               n_estimators=200, max_depth=3, random_state=42)
    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)

    # ---- Method 2: Conformal prediction ----
    # Step 1: train a normal model on the training countries
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    # Step 2: see how far off it is on the calibration countries
    cal_preds = model.predict(X_cal)
    errors = np.abs(y_cal.values - cal_preds)

    # Step 3: the margin is the error that 95% of predictions stay within
    alpha = 0.05                                      # 1 - 0.95 confidence
    n = len(errors)
    q_level = np.ceil((1 - alpha) * (n + 1)) / n      # a small correction for the sample size
    margin = np.quantile(errors, min(q_level, 1.0))
    print(f'Conformal margin: +/- {margin:.1f} points')

    # Step 4: apply both methods to every country in the latest year
    latest = df[df['year'] == latest_year].copy()
    X_latest = latest[feature_cols]
    predicted = model.predict(X_latest)

    bounds = pd.DataFrame({
        'country': latest['country'].values,
        'country_name': latest['country_name'].values,
        'true_value': latest['economic_score'].values,
        'predicted': predicted,
        'lower_quantile': lower_model.predict(X_latest),
        'upper_quantile': upper_model.predict(X_latest),
        'lower_conformal': predicted - margin,
        'upper_conformal': predicted + margin,
        'used_for_calibration': latest['country'].isin(cal_countries).values
    })

    # Check: how many calibration countries' true score is inside the range?
    # It should be close to 95%.
    cal = bounds[bounds['used_for_calibration']]
    inside = (cal['true_value'] >= cal['lower_conformal']) & (cal['true_value'] <= cal['upper_conformal'])
    print(f'True score inside the conformal range for {inside.mean() * 100:.0f}% of calibration countries')
    return bounds


def copy_to_results():
    os.makedirs('results', exist_ok=True)   # exist_ok=True: no crash if it already exists

    files_to_save = [
        'economic_data.csv',        # Part 2
        'xgb_scores.csv',           # Part 4
        'xgb_model.json',           # Part 4
        'shap_scores.csv',          # Part 5
        'shap_summary_plot.png',    # Part 5
        'llm_scores.csv',           # Part 6
        'comparison.csv',           # Part 8
        'comparison_metrics.csv',   # Part 8
        'comparison_chart.png',     # Part 8
        'attribution_heatmap.png',  # Part 8
        'prediction_scatter.png',   # Part 8
        'conformal_bounds.csv',     # Part 10
        'strategy_plans.csv',       # strategy_planner.py all
        'strategy_ratings.csv',     # strategy_planner.py all
        'environmental_cost.csv',   # environmental_cost.py
        'environmental_chart.png',  # environmental_cost.py
        'scoreboard.csv',           # scoreboard.py (the final result)
        'scoreboard.png'            # scoreboard.py
    ]
    for f in files_to_save:
        if os.path.exists(f):
            shutil.copy(f, os.path.join('results', f))
            print('Saved:', f)
        else:
            print('Missing:', f, '- run the earlier parts first')


if __name__ == '__main__':
    df = pd.read_csv('economic_data.csv')
    df = make_score(df)

    bounds = compute_bounds(df)
    bounds.to_csv('conformal_bounds.csv', index=False)
    print('Saved conformal_bounds.csv')

    copy_to_results()
