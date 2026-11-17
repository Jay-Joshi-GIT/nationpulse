# shap_explainer.py  (Part 5 of the NationPulse guide)
#
# SHAP explains WHY the XGBoost model gave each country its score: how much
# each indicator pushed the score up or down.
#
# Input:  xgb_model.json + economic_data.csv
# Output: shap_scores.csv        (one row per country, one column per indicator)
#         shap_summary_plot.png  (all indicators ranked by importance)
#
# Run:  python shap_explainer.py

import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')            # draw plots into files, not windows
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from xgboost_scorer import feature_cols, latest_year


def compute_shap(model, X, feature_cols, countries):
    # TreeExplainer is the SHAP explainer made for tree models like XGBoost
    explainer = shap.TreeExplainer(model)

    # One row per country, one column per indicator.
    # A value of +5 means "this indicator pushed the score UP by 5 points".
    shap_vals = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_vals, columns=feature_cols)
    shap_df.insert(0, 'country', countries)
    shap_df.to_csv('shap_scores.csv', index=False)
    print('Saved shap_scores.csv')

    shap.summary_plot(shap_vals, X, show=False)   # show=False: don't open a window
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=100)
    plt.close()
    print('Saved shap_summary_plot.png')

    return shap_vals


def explain_one_country(shap_vals, row_number, country_name):
    print('Why', country_name, 'got its score:')
    for feature, value in zip(feature_cols, shap_vals[row_number]):
        direction = 'up' if value > 0 else 'down'
        print(f'  {feature}: pushed score {direction} by {abs(value):.2f}')


if __name__ == '__main__':
    model = XGBRegressor()
    model.load_model('xgb_model.json')

    df = pd.read_csv('economic_data.csv')
    latest = df[df['year'] == latest_year].reset_index(drop=True)

    shap_vals = compute_shap(model, latest[feature_cols], feature_cols, latest['country'])

    # Example: print the explanation for the first country
    explain_one_country(shap_vals, 0, latest['country_name'][0])