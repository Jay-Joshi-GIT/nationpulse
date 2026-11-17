# strategy_planner.py  —  future score and strategies to reach a target
#
# Given a country, a target score and a number of years, this file:
#
#   1. Forecasts the score in that many years if current trends simply
#      continue ("business as usual").
#   2. System 1 (XGBoost + SHAP): searches for the smallest realistic changes
#      to the indicators that the model says would reach the target, moving
#      each indicator only the way standard economics recommends. SHAP then
#      splits the score gain between the indicators, so we can see how much
#      each change contributes.
#      The search is also run with no economics rule, to show where the model
#      on its own would recommend the "wrong" way (a weakness of System 1).
#   3. System 2 (LLM): asks Mistral for its own plan: the value each indicator
#      should reach, a priority and a short reason.
#   4. Checks all plans the same way, without using either system as the
#      judge (each out of 100):
#        - Realism:   share of the changes within what countries have actually achieved
#        - Direction: share of the changes that go the way standard economics
#                     recommends (for example lower unemployment)
#      Whether a plan would really reach the target is NOT checked: only a
#      model could say that, and the model would be one of the competitors.
#
# Important: XGBoost and SHAP describe what the DATA associates with higher
# scores. That is not proof that changing an indicator CAUSES a higher score.
#
# Input:  economic_data.csv, xgb_model.json  (run xgboost_scorer.py first)
#         Ollama running with Mistral (for System 2)
# Output (only when run for all countries): strategy_plans.csv, strategy_ratings.csv
#
# Run for one country:   python strategy_planner.py India 50 5
#                        (country name, target score, years)
# Run for all countries: python strategy_planner.py all
# Re-check saved plans:  python strategy_planner.py recheck   (no LLM needed)

import os
import sys
import json
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from xgboost_scorer import feature_cols, latest_year, make_score
from llm_scorer import ask_llm


# ----- Load the data and the trained model once -----

df = pd.read_csv('economic_data.csv')
df = make_score(df)

model = XGBRegressor()
model.load_model('xgb_model.json')
explainer = shap.TreeExplainer(model)

first_year = df['year'].min()


# ----- How fast can each indicator realistically change? -----
#
# For every country we measure the average change per year between the first
# and the last year of data. The pace managed by the fastest 10% of countries
# is our limit for "realistic". Example: life expectancy rose by at most about
# 0.4 years per year, so a 5-year plan may raise it by about 2 years.

def realistic_change_per_year():
    first = df[df['year'] == first_year].set_index('country')[feature_cols]
    last = df[df['year'] == latest_year].set_index('country')[feature_cols]
    per_year = (last - first).abs() / (latest_year - first_year)
    return per_year.quantile(0.9)

max_change_per_year = realistic_change_per_year()

# Plans must also stay inside the range of values the model has seen,
# because a tree model cannot say anything about values outside it.
lowest_seen = df[feature_cols].min()
highest_seen = df[feature_cols].max()


# ----- Which way is "better" for each indicator? -----
#
# The direction standard economics recommends: +1 means "raise it",
# -1 means "lower it". The main XGBoost plan may only move indicators this way.
# The model alone sometimes prefers the opposite (for example more debt,
# because many rich countries have high debt). That is a correlation in the
# data, not good advice, so we report those moves as a weakness of the
# explainable system instead of recommending them.

good_direction = {
    'inflation': -1,
    'unemployment': -1,
    'foreign_direct_investment': 1,
    'government_debt': -1,
    'trade_openness': 1,
    'human_development': 1,
    'education_spending': 1
}

# Lower inflation is better only down to about 2% (the usual central bank
# target); falling prices (deflation) are not a goal.
lowest_inflation_goal = 2.0


# ----- Turning a score into GDP per capita (to make scores easier to read) -----
# This reverses the formula in make_score().

log_gdp = np.log(df['gdp_per_capita'])

def score_to_gdp(score):
    return np.exp(log_gdp.min() + score / 100 * (log_gdp.max() - log_gdp.min()))


# ----- Small helpers -----

def predict_score(values):
    # values is a dictionary like {'inflation': 5.2, 'unemployment': 7.1, ...}
    row = pd.DataFrame([values])[feature_cols]
    return float(model.predict(row)[0])


def get_country(country_name):
    # The country's row for the latest year
    rows = df[(df['country_name'] == country_name) & (df['year'] == latest_year)]
    return rows.iloc[0]


# ----- Step 1: Forecast if nothing changes -----
#
# Each indicator keeps changing at its average speed of the last 10 years.

def forecast_values(country_name, years):
    history = df[df['country_name'] == country_name].sort_values('year')
    first = history.iloc[0]
    last = history.iloc[-1]

    future = {}
    for col in feature_cols:
        change_per_year = (last[col] - first[col]) / (last['year'] - first['year'])
        value = last[col] + change_per_year * years
        # np.clip keeps the value inside the range the model has seen
        future[col] = float(np.clip(value, lowest_seen[col], highest_seen[col]))
    return future


# ----- Step 2: System 1, the XGBoost plan -----
#
# A step-by-step search. In each round we try moving every indicator by 10%,
# 20%, ... 100% of its realistic limit, ask the model for the new score, and
# keep the single move that gains the most score per step of effort. We stop
# when the target is reached, or when no move helps any more.
#
# conventional_only=True  -> indicators only move in their good direction
#                            (this is the main System 1 plan)
# conventional_only=False -> the model may move them either way
#                            (used to show where the model disagrees with economics)

def xgboost_plan(start, target_model_score, years, conventional_only=True):
    plan = dict(start)
    score = predict_score(plan)

    for round_number in range(60):
        if score >= target_model_score:
            break

        # List every move we could make in this round
        candidates = []    # each item: (indicator, new value, size of the move in steps)
        for col in feature_cols:
            if pd.isna(plan[col]):
                continue                     # no data for this indicator
            limit = max_change_per_year[col] * years
            step = limit / 10
            if conventional_only:
                directions = [good_direction[col]]
            else:
                directions = [1, -1]
            for direction in directions:
                for size in range(1, 11):
                    new_value = plan[col] + direction * size * step
                    too_far = abs(new_value - start[col]) > limit + 1e-9
                    out_of_range = new_value < lowest_seen[col] or new_value > highest_seen[col]
                    below_inflation_goal = (conventional_only and col == 'inflation'
                                            and direction == -1 and new_value < lowest_inflation_goal)
                    if too_far or out_of_range or below_inflation_goal:
                        break
                    candidates.append((col, new_value, size))

        if not candidates:
            break

        # Ask the model to score all candidate moves at once
        rows = []
        for col, new_value, size in candidates:
            row = dict(plan)
            row[col] = new_value
            rows.append(row)
        new_scores = model.predict(pd.DataFrame(rows)[feature_cols])

        # Keep the move with the biggest gain per step of effort
        best_move = None
        best_gain_per_step = 0
        for (col, new_value, size), new_score in zip(candidates, new_scores):
            gain_per_step = (new_score - score) / size
            if gain_per_step > best_gain_per_step:
                best_gain_per_step = gain_per_step
                best_move = (col, new_value, float(new_score))

        if best_move is None:
            break                            # no move raises the score any more

        col, new_value, new_score = best_move
        plan[col] = new_value
        score = new_score

    return plan


def shap_contributions(start, plan):
    # SHAP splits a score exactly between the indicators, so the change in
    # each indicator's SHAP value is that indicator's share of the score gain.
    # The shares always add up to the total gain. An indicator the plan does
    # not change can still get a small share, because in the model the effect
    # of one indicator depends on the others (they "interact").
    rows = pd.DataFrame([start, plan])[feature_cols]
    shap_vals = explainer.shap_values(rows)
    contributions = {}
    for i, col in enumerate(feature_cols):
        contributions[col] = float(shap_vals[1][i] - shap_vals[0][i])
    return contributions


# ----- Step 3: System 2, the LLM plan -----

def build_strategy_prompt(country_name, start, current_score, target, years):
    lines = [
        f'Country: {country_name}',
        f'Its economic score is {current_score:.0f} out of 100 '
        f'(GDP per capita about ${score_to_gdp(current_score):,.0f}).',
        f'Goal: reach a score of {target:.0f} '
        f'(GDP per capita about ${score_to_gdp(target):,.0f}) within {years} years.',
        'The score is based on GDP per capita: 0 is the poorest and 100 the richest of 52 countries.',
        '',
        'Current indicators:'
    ]
    known_cols = [col for col in feature_cols if not pd.isna(start[col])]
    for col in known_cols:
        lines.append(f'{col}: {start[col]:.2f}')
    lines.append('')
    lines.append(f'Propose a realistic plan: the value each indicator should reach in {years} years,')
    lines.append('a priority for each indicator (1 = most important) and a reason of at most 10 words.')
    # Placeholders, not example numbers, so the LLM is not anchored on them
    lines.append('Reply ONLY as JSON in this format, with your own values in place of the <...>:')
    example = ', '.join(
        f'"{col}": {{"target_value": <number>, "priority": <rank>, "reason": "<short reason>"}}'
        for col in known_cols)
    lines.append('{"plan": {' + example + '}}')
    return '\n'.join(lines)


def parse_llm_plan(text, start):
    try:
        items = json.loads(text.strip())['plan']
    except Exception:
        return None, {}, {}                  # the reply was not usable

    plan = dict(start)                       # indicators the LLM skips stay unchanged
    priorities = {}
    reasons = {}
    for col in feature_cols:
        if pd.isna(start[col]) or col not in items:
            continue
        try:
            plan[col] = float(items[col]['target_value'])
        except Exception:
            continue
        try:
            priorities[col] = float(items[col]['priority'])
        except Exception:
            pass
        reasons[col] = str(items[col].get('reason', ''))
    return plan, priorities, reasons


# ----- Step 4: Check every plan the same way -----
#
# No system judges another. Every plan gets the same two checks, which need
# no model at all:
#   - Realism:   share of its changes no faster than the fastest 10% of
#                countries managed, and inside the range of values seen
#   - Direction: share of its changes that go the way standard economics
#                recommends (for example lower unemployment)

# The saved plan tables round numbers to 2 decimals, so allow that much slack
rounding_tolerance = 0.01


def change_direction(start, plan, col):
    # 'unchanged', 'as recommended' or 'against economics'
    change = plan[col] - start[col]
    if pd.isna(start[col]) or abs(change) < 1e-6:
        return 'unchanged'
    if change * good_direction[col] > 0:
        return 'as recommended'
    return 'against economics'


def is_realistic(start, plan, col, years):
    limit = max_change_per_year[col] * years
    within_limit = abs(plan[col] - start[col]) <= limit + rounding_tolerance
    within_range = (lowest_seen[col] - rounding_tolerance <= plan[col]
                    <= highest_seen[col] + rounding_tolerance)
    return within_limit and within_range


def against_economics(start, plan):
    # Indicators the plan moves the opposite way to good_direction
    return [col for col in feature_cols
            if change_direction(start, plan, col) == 'against economics']


def rate_plan(start, plan, years):
    changed = [col for col in feature_cols if change_direction(start, plan, col) != 'unchanged']
    realistic = [col for col in changed if is_realistic(start, plan, col, years)]
    wrong_way = against_economics(start, plan)
    return {
        'changes': len(changed),
        'unrealistic_changes': len(changed) - len(realistic),
        'against_economics': ', '.join(wrong_way),
        'realism': 100.0 * len(realistic) / len(changed) if changed else np.nan,
        'direction': 100.0 * (len(changed) - len(wrong_way)) / len(changed) if changed else np.nan
    }


# ----- Putting it all together -----

def plan_table(start, plan, years, contributions=None, priorities=None, reasons=None):
    # contributions: SHAP score gain per indicator (only for XGBoost's own plans)
    # priorities, reasons: what the LLM said about each indicator
    rows = []
    for col in feature_cols:
        direction = change_direction(start, plan, col)
        row = {
            'indicator': col,
            'current': round(start[col], 2),
            'proposed': round(plan[col], 2),
            'change': round(plan[col] - start[col], 2),
            'realistic_limit': round(max_change_per_year[col] * years, 2),
            'realistic': '' if direction == 'unchanged' else
                         ('yes' if is_realistic(start, plan, col, years) else 'no'),
            'economic_direction': direction
        }
        if contributions is not None:
            row['score_contribution'] = round(contributions[col], 2)
        if priorities is not None:
            row['priority'] = priorities.get(col)
            row['reason'] = reasons.get(col, '')
        rows.append(row)
    return pd.DataFrame(rows)


def make_plans(country_name, target, years):
    country = get_country(country_name)
    start = {col: country[col] for col in feature_cols}
    current_score = float(country['economic_score'])

    # The model's estimate for a country can be a little off. We keep that
    # error ("offset") fixed and ask the model for the same increase in score.
    offset = current_score - predict_score(start)
    target_model_score = target - offset

    forecast = predict_score(forecast_values(country_name, years)) + offset

    result = {
        'country': country['country'],
        'country_name': country_name,
        'income_level': country['income_level'],
        'region': country['region'],
        'current_score': current_score,
        'target': target,
        'years': years,
        'forecast': forecast,
        'note': None
    }

    # If the country is already at or above the target, there is nothing to
    # plan. But current trends could still pull it below the target later.
    if target <= current_score:
        if forecast >= target:
            result['note'] = (f'{country_name} is already at or above the target, and if current '
                              f'trends continue it stays above it ({forecast:.1f} in {years} years). '
                              f'No plan is needed. Choose a target above {current_score:.1f} to get plans.')
        else:
            result['note'] = (f'{country_name} is above the target now, but if current trends continue '
                              f'it falls to {forecast:.1f} in {years} years, below the target. '
                              f'Choose a target above {current_score:.1f} to get plans for improving.')
        return result

    # System 1: the main plan, moving indicators only in their good direction
    xgb_plan = xgboost_plan(start, target_model_score, years, conventional_only=True)
    result['xgb_plan_score'] = predict_score(xgb_plan) + offset     # XGBoost's own estimate
    result['xgb_table'] = plan_table(start, xgb_plan, years, shap_contributions(start, xgb_plan))
    result['xgb_rating'] = rate_plan(start, xgb_plan, years)

    # System 1 without the economics rule: the model's own choice
    free_plan = xgboost_plan(start, target_model_score, years, conventional_only=False)
    result['free_plan_score'] = predict_score(free_plan) + offset
    result['free_table'] = plan_table(start, free_plan, years, shap_contributions(start, free_plan))
    result['free_rating'] = rate_plan(start, free_plan, years)

    # System 2
    try:
        text = ask_llm(build_strategy_prompt(country_name, start, current_score, target, years))
        llm_plan, llm_priorities, llm_reasons = parse_llm_plan(text, start)
    except Exception as error:
        print('Could not get a plan from the LLM:', error)
        llm_plan = None

    if llm_plan is None:
        result['llm_table'] = None
        result['llm_rating'] = None
    else:
        result['llm_table'] = plan_table(start, llm_plan, years,
                                         priorities=llm_priorities, reasons=llm_reasons)
        result['llm_rating'] = rate_plan(start, llm_plan, years)
    return result


def reached_text(plan_score, target):
    if plan_score >= target - 0.05:
        return 'reaches the target'
    return 'falls short of the target'


def rating_text(rating):
    if rating is None:
        return 'no usable plan'
    text = (f"{rating['changes']} changes | Realism {rating['realism']:.0f} | "
            f"Direction {rating['direction']:.0f}")
    if rating['against_economics']:
        text += ' | against economics: ' + rating['against_economics']
    return text


def print_result(result):
    print(f"\n{result['country_name']}: score {result['current_score']:.1f} now "
          f"(GDP per capita about ${score_to_gdp(result['current_score']):,.0f})")
    print(f"XGBoost forecast if current trends continue, in {result['years']} years: {result['forecast']:.1f}")
    print(f"Target: {result['target']:.1f} (GDP per capita about ${score_to_gdp(result['target']):,.0f})")

    if result['note'] is not None:
        print('\n' + result['note'])
        return

    print(f"\nSystem 1 (XGBoost + SHAP), economics rule -> XGBoost estimates "
          f"{result['xgb_plan_score']:.1f} ({reached_text(result['xgb_plan_score'], result['target'])})")
    print(result['xgb_table'].to_string(index=False))
    print('  ' + rating_text(result['xgb_rating']))

    print(f"\nSystem 1, model's own choice -> XGBoost estimates {result['free_plan_score']:.1f}")
    print('  ' + rating_text(result['free_rating']))

    print('\nSystem 2 (LLM)')
    if result['llm_table'] is not None:
        print(result['llm_table'].to_string(index=False))
    print('  ' + rating_text(result['llm_rating']))


# Names used for the three plans in the saved files
systems = {'xgboost': 'xgb', 'xgboost_no_economics_rule': 'free', 'llm': 'llm'}


def rating_columns(rating, prefix):
    # e.g. {'llm_realism': 80, 'llm_direction': 100, ...}
    if rating is None:
        return {}
    return {prefix + '_' + key: value for key, value in rating.items()}


def summarise(ratings):
    print('\nAverage checks by income level (each out of 100):')
    columns = []
    for prefix in systems.values():
        columns += [prefix + '_realism', prefix + '_direction']
    print(ratings.groupby('income_level')[columns].mean().round(1).to_string())
    print('\nAll countries:')
    print(ratings[columns].mean().round(1).to_string())
    for prefix in systems.values():
        wrong = (ratings[prefix + '_against_economics'].fillna('') != '').sum()
        print(f'{prefix}: plans with a move against economics in {wrong} of {len(ratings)} countries')


def save_batch(plan_rows, rating_rows):
    if plan_rows:
        pd.concat(plan_rows).to_csv('strategy_plans.csv', index=False)
    pd.DataFrame(rating_rows).to_csv('strategy_ratings.csv', index=False)


def plan_all_countries(score_increase=10, years=5):
    # For the report: every country aims for 10 points more in 5 years.
    # The files are saved after every country, so if the run stops halfway,
    # running it again skips the countries that are already done.
    plan_rows = []
    rating_rows = []
    done = []
    if os.path.exists('strategy_ratings.csv'):
        old_ratings = pd.read_csv('strategy_ratings.csv')
        rating_rows = old_ratings.to_dict('records')
        if os.path.exists('strategy_plans.csv'):
            plan_rows = [pd.read_csv('strategy_plans.csv')]
        done = list(old_ratings['country'])
        print(f'{len(done)} countries already done, skipping them')

    latest = df[df['year'] == latest_year].sort_values('country_name')

    for idx, country in latest.iterrows():
        if country['country'] in done:
            continue
        target = min(country['economic_score'] + score_increase, 100)
        result = make_plans(country['country_name'], target, years)
        print_result(result)

        row = {key: result[key] for key in ['country', 'country_name', 'income_level', 'region',
                                            'current_score', 'target', 'forecast', 'note']}

        # Already at the top of the data (score 100): no plans to make
        if result['note'] is None:
            row['xgb_plan_score'] = result['xgb_plan_score']
            row['free_plan_score'] = result['free_plan_score']
            for system, prefix in systems.items():
                table = result[prefix + '_table']
                if table is not None:
                    table = table.copy()
                    table.insert(0, 'system', system)
                    table.insert(0, 'country', result['country'])
                    plan_rows.append(table)
                row.update(rating_columns(result[prefix + '_rating'], prefix))
        rating_rows.append(row)
        save_batch(plan_rows, rating_rows)

    print('\nSaved strategy_plans.csv and strategy_ratings.csv')
    ratings = pd.DataFrame(rating_rows)
    summarise(ratings[ratings['note'].isna()])


def rate_saved_plans(years=5):
    # Re-checks the plans already in strategy_plans.csv without asking the LLM
    # again (useful when the checks change). Rewrites both files.
    plans = pd.read_csv('strategy_plans.csv')
    old = pd.read_csv('strategy_ratings.csv')
    keep = ['country', 'country_name', 'income_level', 'region', 'current_score',
            'target', 'forecast', 'note', 'xgb_plan_score', 'free_plan_score']
    rating_rows = old[keep].to_dict('records')

    new_tables = []
    for row in rating_rows:
        for system, prefix in systems.items():
            rows = plans[(plans['country'] == row['country']) & (plans['system'] == system)]
            if rows.empty:
                continue
            start = dict(zip(rows['indicator'], rows['current']))
            plan = dict(zip(rows['indicator'], rows['proposed']))
            row.update(rating_columns(rate_plan(start, plan, years), prefix))

            # Rebuild the table with the new columns, keeping what each system said
            table = plan_table(start, plan, years)
            if system != 'llm':
                table['score_contribution'] = rows['score_contribution'].values
            else:
                table['priority'] = rows['priority'].values
                table['reason'] = rows['reason'].values
            table.insert(0, 'system', system)
            table.insert(0, 'country', row['country'])
            new_tables.append(table)

    save_batch(new_tables, rating_rows)
    print('Re-checked the saved plans and rewrote strategy_plans.csv and strategy_ratings.csv')
    ratings = pd.DataFrame(rating_rows)
    summarise(ratings[ratings['note'].isna()])


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'all':
        plan_all_countries()
    elif len(sys.argv) == 2 and sys.argv[1] == 'recheck':
        rate_saved_plans()
    elif len(sys.argv) == 4:
        # e.g.  python strategy_planner.py India 50 5
        result = make_plans(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))
        print_result(result)
    else:
        print('Use:  python strategy_planner.py <country name> <target score> <years>')
        print('  or: python strategy_planner.py all      (all countries, asks the LLM)')
        print('  or: python strategy_planner.py recheck  (re-check saved plans, no LLM)')
        print('Country names with spaces need quotes, e.g. "South Korea"')
