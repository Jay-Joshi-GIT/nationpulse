# environmental_cost.py  —  estimated energy, CO2 and water used by each system
#
# Estimates the electricity each system needs for the two jobs in this
# project (52 countries each), and turns it into CO2 and water:
#
#   1. Scoring + explanation  - System 1: train XGBoost, score, SHAP
#                               System 2: the LLM prompts in llm_scorer.py
#   2. Strategy plans         - System 1: the two greedy plans + SHAP
#                               System 2: the strategy prompt in strategy_planner.py
#
# How: energy = run time x the CPU's rated power. These are ESTIMATES, not
# measurements: they assume the CPU works at its rated power for the whole run.
#   - System 1 is fast, so its run time is measured here, on all 52 countries.
#   - System 2 takes over an hour per job, so its run times are the recorded
#     times of the real runs (see llm_run_seconds below), not run again.
#
# Input:  economic_data.csv, xgb_model.json
# Output: environmental_cost.csv   (one row per job and system)
#         environmental_chart.png  (energy, CO2 and water, side by side)
#
# Run:  python environmental_cost.py

import time
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from xgboost_scorer import feature_cols, latest_year, make_score
import strategy_planner as planner

# ----- The numbers behind the estimate (change them here) -----

# Power: this laptop's Intel Core i7-10610U is rated at 15 W of sustained
# power (TDP); laptop makers can set it anywhere from 12.5 to 25 W.
# Source: Intel's specification page and Notebookcheck. This is the processor
# only; the rest of the laptop used about 6 W at rest (measured 2026-11-13).
cpu_watts = 15
cpu_watts_low = 12.5
cpu_watts_high = 25

# CO2: average for the Indian grid, FY 2023-24, from the Central Electricity
# Authority's CO2 Baseline Database, version 20 (0.727 tonnes per MWh).
kg_co2_per_kwh = 0.727

# Water: power plants use water for cooling. 3.5 litres per kWh is the legal
# limit for Indian coal plants (2015 rules); studies find 1.7 to 8. A laptop
# itself uses no cooling water, so this is the only water counted.
litres_water_per_kwh = 3.5

# Run times of the real LLM runs, in seconds, from the start and end times
# recorded in the project's Claude Code session logs:
#   llm_scorer.py (52 countries x 3 prompts):  2026-10-20, 08:22:28 to 09:40:47 UTC
#   strategy_planner.py all (51 LLM plans):    2026-11-04, 08:14:08 to 09:39:05 UTC
llm_run_seconds = {
    'Scoring + explanation': 4699,
    'Strategy plans': 5097
}


# ----- Timing System 1 -----

df = make_score(pd.read_csv('economic_data.csv'))
latest = df[df['year'] == latest_year].reset_index(drop=True)


def xgboost_scoring():
    # The same work as xgboost_scorer.py + shap_explainer.py, without saving files:
    # train on 80% of countries, test, train again on all, score, explain.
    countries = sorted(df['country'].unique())
    train_countries, test_countries = train_test_split(countries, test_size=0.2, random_state=42)
    is_train = df['country'].isin(train_countries)
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(df.loc[is_train, feature_cols], df.loc[is_train, 'economic_score'])
    model.predict(df.loc[~is_train, feature_cols])
    model.fit(df[feature_cols], df['economic_score'])
    model.predict(latest[feature_cols])
    shap.TreeExplainer(model).shap_values(latest[feature_cols])


def xgboost_strategies():
    # The System 1 part of strategy_planner.py for every country: target = +10 in 5 years
    for idx, country in latest.iterrows():
        start = {col: country[col] for col in feature_cols}
        target = min(country['economic_score'] + 10, 100)
        if target <= country['economic_score']:
            continue      # already at the top (no plan is made in the batch either)
        offset = country['economic_score'] - planner.predict_score(start)
        for conventional_only in [True, False]:
            plan = planner.xgboost_plan(start, target - offset, 5, conventional_only)
            planner.shap_contributions(start, plan)


def average_seconds(task, repeats=5):
    # Runs the task a few times and returns the average time of one run
    start = time.time()
    for i in range(repeats):
        task()
    return (time.time() - start) / repeats


# ----- Turning time into energy, CO2 and water -----

def estimate(job, system, seconds):
    energy_wh = seconds * cpu_watts / 3600        # watt-hours = watts x hours
    return {
        'job': job,
        'system': system,
        'seconds': seconds,
        'energy_wh': energy_wh,
        'energy_wh_low': seconds * cpu_watts_low / 3600,
        'energy_wh_high': seconds * cpu_watts_high / 3600,
        'co2_g': energy_wh / 1000 * kg_co2_per_kwh * 1000,
        'water_ml': energy_wh / 1000 * litres_water_per_kwh * 1000,
        'method': f'estimate: run time x {cpu_watts} W CPU rating'
    }


if __name__ == '__main__':
    print('Timing System 1 (XGBoost + SHAP) on all 52 countries...')
    rows = [
        estimate('Scoring + explanation', 'XGBoost + SHAP', average_seconds(xgboost_scoring)),
        estimate('Strategy plans', 'XGBoost + SHAP', average_seconds(xgboost_strategies, repeats=2))
    ]
    for job, seconds in llm_run_seconds.items():
        rows.append(estimate(job, 'LLM (Mistral 7B)', seconds))

    results = pd.DataFrame(rows)
    results.to_csv('environmental_cost.csv', index=False)
    print('Saved environmental_cost.csv')
    print(results[['job', 'system', 'seconds', 'energy_wh', 'co2_g', 'water_ml']].round(4).to_string(index=False))

    # How many times more does the LLM use for the same job?
    for job in llm_run_seconds:
        both = results[results['job'] == job].set_index('system')['energy_wh']
        ratio = both['LLM (Mistral 7B)'] / both['XGBoost + SHAP']
        print(f'{job}: the LLM uses about {ratio:,.0f} times the energy of XGBoost + SHAP')

    # Chart: one panel each for energy, CO2 and water. A log scale is used
    # because the two systems differ by several powers of ten.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [('energy_wh', 'Energy (Wh)'), ('co2_g', 'CO2 (grams)'), ('water_ml', 'Water (millilitres)')]
    jobs = list(llm_run_seconds)
    x = np.arange(len(jobs))
    width = 0.4
    for ax, (column, label) in zip(axes, panels):
        for shift, system in [(-width / 2, 'XGBoost + SHAP'), (width / 2, 'LLM (Mistral 7B)')]:
            values = [results.loc[(results['job'] == job) & (results['system'] == system), column].sum()
                      for job in jobs]
            ax.bar(x + shift, values, width, label=system)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(jobs)
        ax.set_title(label + ' for 52 countries')
    axes[0].legend(loc='upper left')
    fig.suptitle(f'Estimated environmental cost of each system (run time x {cpu_watts} W; '
                 f'{kg_co2_per_kwh} kg CO2 and {litres_water_per_kwh} L water per kWh)')
    plt.tight_layout()
    plt.savefig('environmental_chart.png', dpi=100)
    plt.close()
    print('Saved environmental_chart.png')
