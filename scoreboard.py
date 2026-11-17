# scoreboard.py  —  the final result: which system does better at what?
#
# Collects the results of every comparison into one table, with both systems
# side by side and the better one named for each measure:
#   System 1: XGBoost + SHAP      System 2: the LLM (Mistral)
# Neither system is used to judge the other. The referees are the real data,
# the pace countries have actually managed, standard economics, and cost.
#
# Some rows have no winner, either because only one system can be measured
# (e.g. XGBoost cannot be tested with sentences) or because there is no right
# answer to compare with (e.g. which explanation is true).
#
# Input:  comparison_metrics.csv, conformal_bounds.csv,
#         strategy_ratings.csv, environmental_cost.csv (if it has been run)
# Output: scoreboard.csv  and  scoreboard.png  (the same table as a picture)
#
# Run:  python scoreboard.py   (after all the other files)

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from comparison import winner

rows = []


def add(area, measure, xgb_value, llm_value, better, note=''):
    rows.append({'area': area, 'measure': measure,
                 'xgboost_shap': xgb_value, 'llm': llm_value,
                 'better': winner(xgb_value, llm_value, better) if better else 'no winner',
                 'note': note})


# ----- 1. Scoring countries (from comparison.py) -----
metrics = pd.read_csv('comparison_metrics.csv')
headline = {
    ('accuracy', 'mean_abs_error_points'): 'Accuracy: points from the real score',
    ('ranking', 'spearman_r_with_real_score'): 'Ranking: Spearman r with the real score',
    ('income_fairness', 'worst_group_abs_error'): 'Fairness: worst income group, average error',
    ('region_fairness', 'worst_group_abs_error'): 'Fairness: worst region, average error',
    ('consistency', 'mean_abs_change_points'): 'Consistency: change when asked again',
    ('framing', 'mean_abs_change_points'): 'Framing: change when numbers become sentences',
    ('coverage', 'share_scored'): 'Coverage: share of countries scored',
    ('explanation_agreement', 'mean_spearman_r_shap_vs_llm'): 'Explanations: agreement between SHAP and LLM'
}
for (test, measure), label in headline.items():
    row = metrics[(metrics['test'] == test) & (metrics['measure'] == measure)].iloc[0]
    rows.append({'area': 'Scoring', 'measure': label,
                 'xgboost_shap': row['xgboost_shap'], 'llm': row['llm'],
                 'better': row['better'], 'note': row['note'] if pd.notna(row['note']) else ''})

# ----- 2. Uncertainty (from save_results.py) -----
bounds = pd.read_csv('conformal_bounds.csv')
cal = bounds[bounds['used_for_calibration']]
inside = ((cal['true_value'] >= cal['lower_conformal']) & (cal['true_value'] <= cal['upper_conformal'])).mean()
width = (bounds['upper_conformal'] - bounds['lower_conformal']).mean()
rows.append({'area': 'Scoring', 'measure': 'Uncertainty: gives a 95% range for its score',
             'xgboost_shap': f'yes, +/- {width / 2:.1f} points, holds the real score for {inside:.0%} of unseen countries',
             'llm': 'no', 'better': 'XGBoost + SHAP', 'note': 'Only XGBoost can state how sure it is'})

# ----- 3. Strategy plans (from strategy_planner.py) -----
ratings = pd.read_csv('strategy_ratings.csv')
ratings = ratings[ratings['note'].isna()]       # countries that got plans
add('Strategy', 'Realism: share of changes at a pace countries have achieved',
    round(ratings['xgb_realism'].mean(), 1), round(ratings['llm_realism'].mean(), 1), 'higher',
    'XGBoost searches only within these limits, so it is 100 by design')
add('Strategy', "Direction: share of changes economics recommends (model's own choice)",
    round(ratings['free_direction'].mean(), 1), round(ratings['llm_direction'].mean(), 1), 'higher',
    'With the economics rule added, XGBoost is 100 by design')
add('Strategy', 'Plans with a move against economics (countries)',
    int((ratings['free_against_economics'].fillna('') != '').sum()),
    int((ratings['llm_against_economics'].fillna('') != '').sum()), 'lower',
    f"Out of {len(ratings)} countries; XGBoost without the economics rule")
add('Strategy', 'Coverage: share of countries with a usable plan',
    round(ratings['xgb_changes'].notna().mean(), 3), round(ratings['llm_changes'].notna().mean(), 3), 'higher')
add('Strategy', 'Average number of indicators changed',
    round(ratings['xgb_changes'].mean(), 1), round(ratings['llm_changes'].mean(), 1), None,
    'Fewer changes is simpler, more changes is broader: no winner')
rows.append({'area': 'Strategy', 'measure': 'Explanation of the plan',
             'xgboost_shap': 'SHAP score gain per indicator, adds up exactly to the total',
             'llm': 'priority and a short reason per indicator',
             'better': 'no winner', 'note': 'Different kinds of explanation; neither can be checked against a true answer'})

# ----- 4. Environmental cost (from environmental_cost.py) -----
if os.path.exists('environmental_cost.csv'):
    cost = pd.read_csv('environmental_cost.csv')
    # Time, energy, CO2 and water all come from the same run time, so they
    # count as ONE environmental factor: both jobs added up, shown together,
    # and judged on energy.
    totals = cost.groupby('system')[['seconds', 'energy_wh', 'energy_wh_low', 'energy_wh_high',
                                     'co2_g', 'water_ml']].sum()

    def footprint(system):
        t = totals.loc[system]
        time_text = f"{t['seconds'] / 60:.0f} min" if t['seconds'] >= 60 else f"{t['seconds']:.1f} s"
        return (f"{time_text}, {t['energy_wh']:.3g} Wh, {t['co2_g']:.3g} g CO2, "
                f"{t['water_ml']:.3g} ml water")

    xgb = totals.loc['XGBoost + SHAP']
    llm = totals.loc['LLM (Mistral 7B)']
    rows.append({'area': 'Environment',
                 'measure': 'Environmental cost: scoring + strategy plans, 52 countries (estimated)',
                 'xgboost_shap': footprint('XGBoost + SHAP'), 'llm': footprint('LLM (Mistral 7B)'),
                 'better': winner(xgb['energy_wh'], llm['energy_wh'], 'lower'),
                 'note': (f"{cost['method'].iloc[0]}; the LLM uses about "
                          f"{llm['energy_wh'] / xgb['energy_wh']:,.0f} times the energy "
                          f"({llm['energy_wh_low']:.0f}-{llm['energy_wh_high']:.0f} Wh at 12.5-25 W)")})
else:
    rows.append({'area': 'Environment', 'measure': 'Environmental cost (time, energy, CO2, water)',
                 'xgboost_shap': 'not measured yet', 'llm': 'not measured yet', 'better': 'no winner',
                 'note': 'Run python environmental_cost.py'})

board = pd.DataFrame(rows)
board.to_csv('scoreboard.csv', index=False)
print(board[['area', 'measure', 'xgboost_shap', 'llm', 'better']].to_string(index=False))
print('\nWins:', board['better'].value_counts().to_dict())
print('Saved scoreboard.csv')

# ----- The same table as a picture, with the better system highlighted -----
colours = {'XGBoost + SHAP': '#cfe2f3', 'LLM': '#fce5cd'}
def show(value):
    return 'cannot be tested' if pd.isna(value) else str(value)


cells = [[r['area'], r['measure'], show(r['xgboost_shap']), show(r['llm']), r['better']] for r in rows]
fig, ax = plt.subplots(figsize=(20, 0.3 * len(cells) + 0.8))
ax.axis('off')
table = ax.table(cellText=cells, colLabels=['Area', 'Measure', 'XGBoost + SHAP', 'LLM', 'Better'],
                 colWidths=[0.07, 0.33, 0.27, 0.2, 0.1], bbox=[0, 0, 1, 0.94], cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(8)
for i, r in enumerate(rows, start=1):
    if r['better'] in colours:
        column = 2 if r['better'] == 'XGBoost + SHAP' else 3
        table[i, column].set_facecolor(colours[r['better']])
        table[i, 4].set_facecolor(colours[r['better']])
ax.set_title('NationPulse: which system does better at what? (shaded = better)', fontsize=12, y=0.97)
plt.tight_layout()
plt.savefig('scoreboard.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved scoreboard.png')
