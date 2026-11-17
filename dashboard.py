# dashboard.py  —  shows all NationPulse results in a web page
#
# Two systems compete: System 1 (XGBoost + SHAP) and System 2 (the LLM).
# Neither judges the other; both are compared with the real World Bank score,
# the pace countries have actually managed, and standard economics.
#
# Three tabs:
#   1. Country results   - both scores next to the real score, each system's
#                          error, and which indicators each system says matter
#   2. Strategy planner  - pick a target score; both systems propose a plan,
#                          and every plan gets the same checks
#   3. Scoreboard        - which system does better at what (scoreboard.csv)
#
# Input:  comparison.csv, conformal_bounds.csv, scoreboard.csv,
#         economic_data.csv, xgb_model.json (run every other file first)
# Run:    python dashboard.py   then open the link it prints (http://127.0.0.1:7860)

import json
import pandas as pd
import gradio as gr
import plotly.graph_objects as go

from xgboost_scorer import feature_cols
from strategy_planner import make_plans, score_to_gdp, reached_text

# Load the results once when the dashboard starts
df_main = pd.read_csv('comparison.csv')
df_bounds = pd.read_csv('conformal_bounds.csv')
scoreboard = pd.read_csv('scoreboard.csv')

df_main = pd.merge(df_main, df_bounds[['country', 'lower_conformal', 'upper_conformal']],
                   on='country', how='left')

country_names = sorted(df_main['country_name'])

# Extra columns for the comparison table
df_main['rephrase_change'] = (df_main['llm_score_rephrased'] - df_main['llm_score']).abs()
df_main['framing_change'] = (df_main['llm_score_words'] - df_main['llm_score']).abs()


def number(value, sign=False):
    # Turns a number into short text, e.g. 12.345 -> "12.3" (or "+12.3" with sign=True)
    if pd.isna(value):
        return 'no data'
    if sign:
        return f'{value:+.1f}'
    return f'{value:.1f}'


def better(xgb_value, llm_value):
    # For errors and changes: the smaller one is better
    if pd.isna(xgb_value) or pd.isna(llm_value):
        return ''
    if abs(xgb_value) < abs(llm_value):
        return 'XGBoost + SHAP'
    if abs(llm_value) < abs(xgb_value):
        return 'LLM'
    return 'tie'


def comparison_table(row):
    # One row per test: both systems for this country, and their averages
    # over all countries
    income_rows = df_main[df_main['income_level'] == row['income_level']]
    tests = [
        ['Error (score minus real score)',
         number(row['xgb_error'], sign=True), number(row['llm_error'], sign=True),
         better(row['xgb_error'], row['llm_error']),
         f"XGBoost {number(df_main['xgb_error'].abs().mean())} / LLM {number(df_main['llm_error'].abs().mean())} points off"],
        ['Average error, ' + row['income_level'],
         number(income_rows['xgb_error'].mean(), sign=True), number(income_rows['llm_error'].mean(), sign=True),
         better(income_rows['xgb_error'].mean(), income_rows['llm_error'].mean()),
         '+ means the system rates this income group too high'],
        ['Change when asked again',
         '0.0 (always the same)', number(row['rephrase_change']),
         better(0, row['rephrase_change']),
         f"LLM {number(df_main['rephrase_change'].mean())} points on average"],
        ['Change when numbers become sentences',
         'cannot be tested', number(row['framing_change']), '',
         f"LLM {number(df_main['framing_change'].mean())} points on average"],
        ['Agreement on which indicators matter',
         f"{row['attribution_agreement']:.2f}", f"{row['attribution_agreement']:.2f}", '',
         f"{df_main['attribution_agreement'].mean():.2f} on average (+1 same order, -1 opposite)"]
    ]
    return pd.DataFrame(tests, columns=['Test', 'XGBoost + SHAP', 'LLM', 'Better here', 'All countries'])


def show_country(country_name):
    row = df_main[df_main['country_name'] == country_name].iloc[0]

    real_text = (f"{row['economic_score']:.1f} / 100   "
                 f"(GDP per capita about ${score_to_gdp(row['economic_score']):,.0f})")
    xgb_text = (f"{row['xgb_score_unseen']:.1f} / 100   (error {row['xgb_error']:+.1f}; "
                f"95% range {row['lower_conformal']:.1f} to {row['upper_conformal']:.1f})")
    if pd.isna(row['llm_score']):
        llm_text = 'The LLM did not give a usable score'
    else:
        llm_text = f"{row['llm_score']:.1f} / 100   (error {row['llm_error']:+.1f}; gives no range)"

    # Bar chart: how much each indicator mattered to each system
    shap_weights = [row['shap_weight_' + col] for col in feature_cols]
    llm_weights = [row['weight_' + col] for col in feature_cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='SHAP (System 1)', x=feature_cols, y=shap_weights))
    fig.add_trace(go.Bar(name='LLM (System 2)', x=feature_cols, y=llm_weights))
    fig.update_layout(barmode='group',
                      title='What drove the score for ' + country_name + ' (each system\'s own view)',
                      yaxis_title='Share of importance (adds up to 1)')

    return real_text, xgb_text, llm_text, comparison_table(row), fig


# ----- Strategy planner tab -----
# Uses strategy_planner.py. This tab asks the LLM live, so Ollama must be
# running, and each plan takes a few minutes on a laptop.

def check_value(rating, key):
    if rating is None:
        return 'no plan'
    value = rating[key]
    if key in ['realism', 'direction']:
        return 'no changes' if pd.isna(value) else round(value)
    return value if value != '' else 'none'


def plan_strategies(country_name, target, years):
    result = make_plans(country_name, float(target), int(years))
    current = result['current_score']

    lines = [
        f"**{country_name}** scores **{current:.1f}** now "
        f"(GDP per capita about ${score_to_gdp(current):,.0f}).",
        f"XGBoost's forecast if current trends continue, in {int(years)} years: **{result['forecast']:.1f}**.",
        f"Target: **{float(target):.1f}** (GDP per capita about ${score_to_gdp(float(target)):,.0f}).",
        ''
    ]

    # Target at or below the current score: no plans, just explain why
    if result['note'] is not None:
        lines.append(result['note'])
        empty = pd.DataFrame()
        return '\n\n'.join(lines), empty, empty, empty, empty, go.Figure()

    lines += [
        f"System 1 (XGBoost + SHAP) says its plan {reached_text(result['xgb_plan_score'], float(target))} "
        f"({result['xgb_plan_score']:.1f}). The LLM gives no estimate for its plan.",
        '',
        '*Whether a plan would really reach the target is not checked: only a model '
        'could say that, and the model is one of the competitors. Every plan gets the '
        'same model-free checks below.*'
    ]
    if result['llm_rating'] is None:
        lines.append('')
        lines.append('System 2 (LLM): no usable plan. Is Ollama running?')

    plans = [('System 1, economics rule', result['xgb_rating']),
             ("System 1, model's own choice", result['free_rating']),
             ('System 2 (LLM)', result['llm_rating'])]
    checks = [('Realism (out of 100)', 'realism', 'Share of changes at a pace countries have actually achieved'),
              ('Direction (out of 100)', 'direction', 'Share of changes that go the way economics recommends'),
              ('Indicators changed', 'changes', ''),
              ('Moved against economics', 'against_economics', '')]
    rows = []
    for label, key, meaning in checks:
        rows.append([label] + [check_value(rating, key) for name, rating in plans] + [meaning])
    checks_table = pd.DataFrame(rows, columns=['Check'] + [name for name, rating in plans] + ['What it checks'])

    lines.append('')
    lines.append('*XGBoost and SHAP show what the data associates with higher scores, '
                 'which is not proof that a change causes a higher score.*')

    # Chart: each change as a share of its realistic limit (1 = the fastest
    # pace the fastest 10% of countries managed). Above 1 = unrealistic.
    fig = go.Figure()
    for name, table in [('System 1 (economics rule)', result['xgb_table']), ('System 2 (LLM)', result['llm_table'])]:
        if table is not None:
            share = (table['change'].abs() / table['realistic_limit']).round(2)
            fig.add_trace(go.Bar(name=name, x=feature_cols, y=share))
    fig.add_hline(y=1, line_dash='dash', annotation_text='realistic limit')
    fig.update_layout(barmode='group', title='Size of each change compared with its realistic limit',
                      yaxis_title='Change / realistic limit')

    llm_table = result['llm_table'] if result['llm_table'] is not None else pd.DataFrame()
    return '\n\n'.join(lines), result['xgb_table'], result['free_table'], llm_table, checks_table, fig


# Shown under each page: the data must be credited (see CREDITS.md)
data_credit = ('Data: World Bank, World Development Indicators (CC BY 4.0), with original data '
               'from the IMF, ILO, UN Population Division, UNESCO Institute for Statistics and '
               'OECD; government debt from the IMF World Economic Outlook. LLM: Mistral 7B '
               '(Mistral AI, Apache 2.0) run with Ollama. No data provider endorses this project. '
               'Full credits and licences: CREDITS.md')

strategy_tab = gr.Interface(
    fn=plan_strategies,
    inputs=[
        gr.Dropdown(choices=country_names, value='India', label='Country'),
        gr.Slider(0, 100, value=45, step=1, label='Target score'),
        gr.Slider(1, 10, value=5, step=1, label='Years to reach it')
    ],
    outputs=[
        gr.Markdown(),
        gr.Dataframe(label='System 1: XGBoost + SHAP plan (moves only the way economics recommends)', wrap=True),
        gr.Dataframe(label="System 1: the model's own choice (no economics rule)", wrap=True),
        gr.Dataframe(label='System 2: LLM plan', wrap=True),
        gr.Dataframe(label='The same checks for every plan', wrap=True),
        gr.Plot(label='Size of each change')
    ],
    submit_btn='Make plans',
    article=data_credit,
    title='NationPulse strategy planner',
    description='Pick a country, a target score and a number of years. Both systems '
                'propose how to get there, and every plan gets the same checks. Takes a few minutes.'
)


results_tab = gr.Interface(
    fn=show_country,
    inputs=gr.Dropdown(choices=country_names, value=country_names[0], label='Country'),
    outputs=[
        gr.Textbox(label='Real score (World Bank data)'),
        gr.Textbox(label='System 1: XGBoost + SHAP (trained without this country)'),
        gr.Textbox(label='System 2: LLM'),
        gr.Dataframe(label='Both systems on each test', wrap=True),
        gr.Plot(label='Indicator importance')
    ],
    submit_btn='Show results',
    article=data_credit,
    title='NationPulse',
    description='Two systems compete to score countries: an explainable XGBoost model '
                'and an LLM. Both are compared with the real World Bank score.'
)

scoreboard_tab = gr.Interface(
    fn=lambda: scoreboard[['area', 'measure', 'xgboost_shap', 'llm', 'better', 'note']],
    inputs=[],
    outputs=gr.Dataframe(label='Which system does better at what?', wrap=True),
    article=data_credit,
    title='Scoreboard',
    description='All results in one table. Press Generate to show it.'
)

# ----- Hover text for every button -----
# Gradio buttons have no tooltip setting, so this small script, added to the
# page, gives each button a "title" (the text a browser shows on hover).
# It runs again whenever the page changes, because Gradio draws buttons later.

button_help = {
    'Country results': 'Both systems\' scores for one country, next to the real score',
    'Strategy planner': 'Ask both systems for a plan to reach a target score',
    'Scoreboard': 'Which system does better at what, over all tests',
    'Show results': 'Show both systems\' scores and tests for the chosen country',
    'Make plans': 'Ask both systems for a plan for this country, target and number of years (takes a few minutes)',
    'Generate': 'Show the table of which system does better at what',
    '↺': 'Put this slider back to its starting value',
    'Clear': 'Reset the inputs and empty the results',
    'Flag': 'Save these inputs and results to the .gradio/flagged folder, e.g. to note a strange result',
    'Use via API': 'How to call this dashboard from your own code instead of the web page',
    'Settings': 'Change the page\'s language, light or dark theme, and zoom',
    'Stop Recording': 'Stop recording your clicks as API code',
    'More tabs': 'Show the tabs that do not fit on the screen',
    # Icon buttons on tables and charts (matched by their built-in label)
    'Copy table data': 'Copy the whole table, to paste into a spreadsheet',
    'Fullscreen': 'Show this table on the full screen',
    'Open cell menu': 'Sort or filter by this column',
    'Download plot as a PNG': 'Save this chart as a picture',
    'Zoom': 'Drag on the chart to zoom into an area',
    'Pan': 'Drag to move the chart around',
    'Box Select': 'Drag a box to highlight some bars',
    'Lasso Select': 'Draw around bars to highlight them',
    'Zoom in': 'Zoom in on the chart',
    'Zoom out': 'Zoom out of the chart',
    'Autoscale': 'Fit the chart to all its data',
    'Reset axes': 'Go back to the original view of the chart'
}

tooltip_script = '''
<script>
const buttonHelp = ''' + json.dumps(button_help) + ''';
function addTooltips() {
    document.querySelectorAll('button').forEach(button => {
        // Buttons with words are matched by their text, icon buttons by their label
        const name = button.innerText.trim() || button.getAttribute('aria-label') || button.dataset.title || '';
        if (buttonHelp[name]) {
            button.title = buttonHelp[name];
        } else if (!button.title && name) {
            button.title = name;
        }
    });
}
new MutationObserver(addTooltips).observe(document.documentElement, {childList: true, subtree: true});
</script>
'''

# Put the three pages side by side as tabs
demo = gr.TabbedInterface([results_tab, strategy_tab, scoreboard_tab],
                          ['Country results', 'Strategy planner', 'Scoreboard'])

if __name__ == '__main__':
    demo.launch(head=tooltip_script)
