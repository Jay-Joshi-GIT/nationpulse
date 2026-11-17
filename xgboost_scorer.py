# xgboost_scorer.py  (Part 4 of the NationPulse guide)  —  System 1
#
# Trains an XGBoost model that gives each country an economic score (0-100)
# from its indicators, and saves the model and the scores.
#
# Input:  economic_data.csv
# Output: xgb_model.json  (the trained model)
#         xgb_scores.csv  (one score per country, for the latest year:
#                          xgb_score        - from the model trained on all countries
#                          xgb_score_unseen - from a model that never saw that
#                                             country, used for the fair contest)
#
# Run:  python xgboost_scorer.py

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ----- Settings shared with the other files -----
# Other files import these with:  from xgboost_scorer import feature_cols, latest_year

# The inputs (features) the model learns from. Both systems (XGBoost and the
# LLM) get exactly these same indicators.
feature_cols = ['inflation', 'unemployment', 'foreign_direct_investment',
                'government_debt', 'trade_openness',
                'human_development', 'education_spending']

# The year we score and compare countries on (the newest year we have)
latest_year = 2023


# ----- Step 1: Make the target score -----
#
# We need a number for the model to predict. We use GDP per capita, the most
# common single measure of how well-off a country's economy is, turned into
# a score from 0 to 100:
#
#   - We take the log first, because GDP per capita ranges from about $500 to
#     $80,000. Without the log, the rich countries would squash everyone else
#     near 0. With it, going from $1,000 to $2,000 counts the same as going
#     from $20,000 to $40,000 (both are "doubling").
#   - Then the lowest value in the data becomes 0 and the highest becomes 100.

def make_score(df):
    log_gdp = np.log(df['gdp_per_capita'])
    lowest = log_gdp.min()
    highest = log_gdp.max()
    df['economic_score'] = (log_gdp - lowest) / (highest - lowest) * 100
    return df


# ----- Step 2: Train and test the model -----

def train_scorer(df):
    X = df[feature_cols]
    y = df['economic_score']

    # Note: we do NOT drop rows with missing values. XGBoost can work with
    # missing values (NaN) by itself, so we keep every country.

    # Split by COUNTRY, not by row. Each country has 10 rows (one per year)
    # that look almost the same. If India 2019 were in training and India 2020
    # in testing, the test would be too easy. So we keep 20% of the countries
    # completely out of training and test on them.
    all_countries = sorted(df['country'].unique())
    train_countries, test_countries = train_test_split(all_countries, test_size=0.2, random_state=42)

    is_train = df['country'].isin(train_countries)
    X_train, y_train = X[is_train], y[is_train]
    X_test, y_test = X[~is_train], y[~is_train]    # ~ means "not"

    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('Tested on', len(test_countries), 'countries the model never saw:', test_countries)
    print(f'MSE: {mse:.2f}   (lower is better)')
    print(f'R2:  {r2:.2f}   (1.0 is perfect, 0.0 is no better than guessing the average)')

    # Now that we know how good it is, train the final model on ALL countries
    model.fit(X, y)
    model.save_model('xgb_model.json')
    print('Model saved to xgb_model.json')
    return model


def unseen_scores(df):
    # A fair contest with the LLM: XGBoost must not have seen the country it
    # scores (the LLM never sees the answer either). For each country we train
    # a model on the other 51 countries and score the left-out one.
    # This is called "leave one country out".
    latest = df[df['year'] == latest_year]
    scores = {}
    for country in latest['country']:
        others = df[df['country'] != country]
        model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(others[feature_cols], others['economic_score'])
        row = latest[latest['country'] == country]
        scores[country] = float(model.predict(row[feature_cols])[0])
    return latest['country'].map(scores)


if __name__ == '__main__':
    df = pd.read_csv('economic_data.csv')
    df = make_score(df)
    model = train_scorer(df)

    # Score every country for the latest year
    latest = df[df['year'] == latest_year].copy()
    latest['xgb_score'] = model.predict(latest[feature_cols])
    latest['xgb_score_unseen'] = unseen_scores(df)
    error = (latest['xgb_score_unseen'] - latest['economic_score']).abs().mean()
    print(f'Scoring countries it never saw, XGBoost is {error:.1f} points off on average')

    columns = ['country', 'country_name', 'region', 'income_level', 'year',
               'economic_score', 'xgb_score', 'xgb_score_unseen']
    latest[columns].to_csv('xgb_scores.csv', index=False)

    print(latest[['country_name', 'economic_score', 'xgb_score']].head())
    print('Saved xgb_scores.csv')