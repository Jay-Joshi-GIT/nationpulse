# comparison.py  (Part 8 of the NationPulse guide)
#
# Two systems compete on scoring countries:
#   System 1: XGBoost + SHAP      System 2: the LLM (Mistral)
# Neither system judges the other. Both are checked against the same referee:
# the real score from World Bank data (economic_score). For a fair contest
# XGBoost is scored on countries it never saw (xgb_score_unseen), since the
# LLM never sees the real score either.
#
#   1. Accuracy        - how far is each system from the real score?
#   2. Ranking         - does each system put countries in the real order?
#   3. Income fairness - is either system too high or too low for some income levels?
#   4. Region fairness - the same, by region
#   5. Consistency     - same numbers, question asked again differently: same score?
#   6. Framing         - "inflation: 8" vs "prices rose 8%": same score?
#   7. Coverage        - did the system give a usable score for every country?
#   (+) Explanations   - do SHAP and the LLM say the same indicators matter?
#                        (no winner: we cannot tell which explanation is right)
#   (+) Baselines      - two simple methods, so the systems' errors can be read in
#                        context: the average score, and a linear regression
#
# Input:  xgb_scores.csv, llm_scores.csv, shap_scores.csv, economic_data.csv, xgb_model.json
# Output: comparison.csv           (one row per country)
#         comparison_metrics.csv   (one row per measure, both systems side by side)
#         comparison_chart.png     (real score, XGBoost and LLM, per country)
#         attribution_heatmap.png  (SHAP weight minus LLM weight)
#         prediction_scatter.png   (each system's score against the real score)
#
# Run:  python comparison.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')            # draw plots into files, not windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

from xgboost_scorer import feature_cols, latest_year, make_score


def shap_to_weights(shap_df):
    # SHAP values can be + or -. To compare with the LLM's weights (which are
    # all positive and add up to 1), we take the size of each SHAP value and
    # divide by the total for that country.
    weights = shap_df[['country']].copy()
    sizes = shap_df[feature_cols].abs()
    totals = sizes.sum(axis=1)
    for col in feature_cols:
        weights['shap_weight_' + col] = sizes[col] / totals
    return weights


def winner(xgb_value, llm_value, better):
    # better = 'lower' or 'higher': which way is good for this measure
    if pd.isna(xgb_value) or pd.isna(llm_value) or better is None:
        return 'no winner'
    if xgb_value == llm_value:
        return 'tie'
    xgb_wins = xgb_value < llm_value if better == 'lower' else xgb_value > llm_value
    return 'XGBoost + SHAP' if xgb_wins else 'LLM'


def add_metric(metrics, test, measure, xgb_value, llm_value, better, note=''):
    metrics.append({'test': test, 'measure': measure,
                    'xgboost_shap': xgb_value, 'llm': llm_value,
                    'better': winner(xgb_value, llm_value, better), 'note': note})
    print(f'   {measure}: XGBoost + SHAP = {xgb_value}, LLM = {llm_value}')


def baseline_errors():
    # Two simple baselines, scored like XGBoost on countries they never saw
    # (leave one country out):
    #   - mean:   always predict the average score of the other 51 countries
    #   - linear: a linear regression on the same 7 indicators. Linear regression
    #             cannot handle missing values, so each gap is filled with the
    #             median of the training countries only.
    data = make_score(pd.read_csv('economic_data.csv'))
    latest = data[data['year'] == latest_year]
    mean_errors = []
    linear_errors = []
    mean_preds = []
    linear_preds = []
    for idx, row in latest.iterrows():
        train = data[data['country'] != row['country']]
        mean_preds.append(train['economic_score'].mean())
        medians = train[feature_cols].median()
        model = LinearRegression().fit(train[feature_cols].fillna(medians), train['economic_score'])
        X = pd.DataFrame([row[feature_cols]]).astype(float).fillna(medians)
        linear_preds.append(float(model.predict(X)[0]))
    real = latest['economic_score'].values
    return real, np.array(mean_preds), np.array(linear_preds)


def compare_systems(df_xgb, df_llm, df_shap):
    df = pd.merge(df_xgb, df_llm, on='country', how='inner')
    df = pd.merge(df, shap_to_weights(df_shap), on='country', how='inner')

    # error > 0: the system rates the country HIGHER than the real score
    # error < 0: the system rates it LOWER
    df['xgb_error'] = df['xgb_score_unseen'] - df['economic_score']
    df['llm_error'] = df['llm_score'] - df['economic_score']

    metrics = []   # each measure adds one row with both systems' values

    # Accuracy and fairness use the countries both systems scored
    scored = df.dropna(subset=['llm_score'])

    # ---- 1. Accuracy ----
    print('1. Accuracy (points away from the real score, lower is better):')
    add_metric(metrics, 'accuracy', 'mean_abs_error_points',
               round(scored['xgb_error'].abs().mean(), 1), round(scored['llm_error'].abs().mean(), 1), 'lower')

    # ---- (+) Baselines ----
    print('+ Baselines on countries they never saw (points away from the real score):')
    real, mean_preds, linear_preds = baseline_errors()
    for name, preds in [('average_score', mean_preds), ('linear_regression', linear_preds)]:
        error = np.abs(preds - real).mean()
        # A ranking makes no sense for the average-score baseline: its prediction
        # only moves because one country is left out each time
        r = spearmanr(real, preds)[0] if name == 'linear_regression' else np.nan
        print(f'   {name}: {error:.1f} points, Spearman r = {r:.3f}')
        metrics.append({'test': 'baseline_' + name, 'measure': 'mean_abs_error_points',
                        'xgboost_shap': np.nan, 'llm': np.nan, 'better': 'no winner',
                        'note': f'{error:.1f} points; Spearman r {r:.3f}'})

    # ---- 2. Ranking ----
    print('2. Ranking (Spearman r with the real score, higher is better):')
    xgb_r, _ = spearmanr(scored['economic_score'], scored['xgb_score_unseen'])
    llm_r, _ = spearmanr(scored['economic_score'], scored['llm_score'])
    add_metric(metrics, 'ranking', 'spearman_r_with_real_score', round(xgb_r, 3), round(llm_r, 3), 'higher')

    # ---- 3 and 4. Fairness ----
    # Average error per group. A fair system has errors near 0 in every group,
    # so we compare the WORST group of each system.
    for test, column in [('income_fairness', 'income_level'), ('region_fairness', 'region')]:
        print(f'{3 if column == "income_level" else 4}. Average error by {column} (+ too high, - too low):')
        by_group = scored.groupby(column)[['xgb_error', 'llm_error']].mean()
        for group, row in by_group.iterrows():
            add_metric(metrics, test, group, round(row['xgb_error'], 1), round(row['llm_error'], 1), None)
        add_metric(metrics, test, 'worst_group_abs_error',
                   round(by_group['xgb_error'].abs().max(), 1), round(by_group['llm_error'].abs().max(), 1), 'lower')

    # ---- 5. Consistency ----
    # XGBoost gives exactly the same answer to the same numbers every time
    # (we check it by predicting twice). The LLM was asked again in other words.
    print('5. Consistency (points the score changes when asked again, lower is better):')
    model = XGBRegressor()
    model.load_model('xgb_model.json')
    data = pd.read_csv('economic_data.csv')
    X = data[data['year'] == latest_year][feature_cols]
    first = model.predict(X)
    second = model.predict(X)
    both = df.dropna(subset=['llm_score', 'llm_score_rephrased'])
    add_metric(metrics, 'consistency', 'mean_abs_change_points',
               round(float(np.abs(first - second).mean()), 1),
               round((both['llm_score'] - both['llm_score_rephrased']).abs().mean(), 1), 'lower',
               'XGBoost always gives the same output for the same input')

    # ---- 6. Framing ----
    print('6. Framing (points the score changes when numbers are written as sentences):')
    both = df.dropna(subset=['llm_score', 'llm_score_words'])
    add_metric(metrics, 'framing', 'mean_abs_change_points', np.nan,
               round((both['llm_score'] - both['llm_score_words']).abs().mean(), 1), 'lower',
               'XGBoost only reads numbers, so it cannot be tested with sentences')

    # ---- 7. Coverage ----
    print('7. Coverage (share of countries with a usable score):')
    add_metric(metrics, 'coverage', 'share_scored',
               round(df['xgb_score_unseen'].notna().mean(), 3), round(df['llm_score'].notna().mean(), 3), 'higher')

    # ---- (+) Explanations ----
    # For each country: do SHAP and the LLM rank the indicators the same way?
    # This shows how much they agree, not who is right.
    agreements = []
    for idx, row in df.iterrows():
        shap_w = [row['shap_weight_' + col] for col in feature_cols]
        llm_w = [row['weight_' + col] for col in feature_cols]
        if pd.isna(llm_w).any():
            agreements.append(None)
        else:
            corr, _ = spearmanr(shap_w, llm_w)
            agreements.append(corr)
    df['attribution_agreement'] = agreements
    average = df['attribution_agreement'].mean()
    print(f'+ Explanations agree (SHAP vs LLM, average Spearman r): {average:.3f}')
    metrics.append({'test': 'explanation_agreement', 'measure': 'mean_spearman_r_shap_vs_llm',
                    'xgboost_shap': round(average, 3), 'llm': round(average, 3), 'better': 'no winner',
                    'note': 'One shared number: how much the two explanations agree'})

    # ---- Save the tables ----
    df.to_csv('comparison.csv', index=False)
    pd.DataFrame(metrics).to_csv('comparison_metrics.csv', index=False)
    print('Saved comparison.csv and comparison_metrics.csv')

    make_bar_chart(df)
    make_scatter(df)
    make_heatmap(df)
    return df


def make_bar_chart(df):
    df = df.sort_values('economic_score')
    x = np.arange(len(df))     # positions 0, 1, 2, ... for each country
    width = 0.27

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - width, df['economic_score'], width, label='Real score (World Bank data)', color='grey')
    ax.bar(x, df['xgb_score_unseen'], width, label='XGBoost + SHAP (country not seen in training)')
    ax.bar(x + width, df['llm_score'], width, label='LLM')
    ax.set_xticks(x)
    ax.set_xticklabels(df['country_name'], rotation=90)
    ax.set_ylabel('Economic score (0-100)')
    ax.set_title('Both systems against the real score, by country (sorted by real score)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('comparison_chart.png', dpi=100)
    plt.close()
    print('Saved comparison_chart.png')


def make_scatter(df):
    # One dot per country. Dots on the dashed line are exactly right; a flat
    # cloud means the system squeezes every country towards the middle.
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharex=True, sharey=True)
    colours = {'Low income': '#d62728', 'Lower middle income': '#ff7f0e',
               'Upper middle income': '#2ca02c', 'High income': '#1f77b4'}
    for ax, column, name in [(axes[0], 'xgb_score_unseen', 'XGBoost + SHAP (country unseen)'),
                             (axes[1], 'llm_score', 'LLM (Mistral 7B)')]:
        for income, colour in colours.items():
            group = df[df['income_level'] == income]
            ax.scatter(group['economic_score'], group[column], s=18, color=colour, label=income)
        ax.plot([0, 100], [0, 100], '--', color='grey', linewidth=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Real score (World Bank data)')
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel('Predicted score')
    axes[1].legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.savefig('prediction_scatter.png', dpi=200)
    plt.close()
    print('Saved prediction_scatter.png')


def make_heatmap(df):
    # Rows = countries, columns = indicators,
    # cell = SHAP weight minus LLM weight. Strong colours = big disagreement.
    table = pd.DataFrame(index=df['country_name'])
    for col in feature_cols:
        table[col] = (df['shap_weight_' + col] - df['weight_' + col]).values

    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(table, cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'label': 'SHAP weight minus LLM weight'})
    ax.set_title('Where SHAP and the LLM disagree on what matters')
    plt.tight_layout()
    plt.savefig('attribution_heatmap.png', dpi=100)
    plt.close()
    print('Saved attribution_heatmap.png')


if __name__ == '__main__':
    df_xgb = pd.read_csv('xgb_scores.csv')
    df_llm = pd.read_csv('llm_scores.csv')
    df_shap = pd.read_csv('shap_scores.csv')
    compare_systems(df_xgb, df_llm, df_shap)
