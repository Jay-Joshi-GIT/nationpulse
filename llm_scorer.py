# llm_scorer.py  (Part 6 of the NationPulse guide)  —  System 2
#
# Asks a local LLM (Mistral, through Ollama) to score each country from 0 to
# 100 using the SAME indicators the XGBoost model uses, and to say how much
# each indicator mattered.
#
# Each country is asked three times, to test the LLM:
#   1. main      - the numbers in a list          -> score + indicator weights
#   2. rephrased - same numbers, question worded differently  (consistency test)
#   3. words     - same numbers written as sentences, e.g. "prices rose 8%"
#                  instead of "inflation: 8"                  (framing test)
#
# Input:  economic_data.csv
# Output: llm_scores.csv
#
# Before running: install Ollama from ollama.com, then run  ollama pull mistral
# Run:  python llm_scorer.py

import json
import pandas as pd
import ollama

from xgboost_scorer import feature_cols, latest_year

llm_model = 'mistral'


# ----- Building the three kinds of prompt -----

def build_prompt(country, row):
    lines = ['Country: ' + country]
    for col in feature_cols:
        lines.append(f'{col}: {row[col]:.2f}')
    lines.append('')
    lines.append('Based on these indicators, give an overall economic health score')
    lines.append('from 0 (very poor) to 100 (excellent).')
    lines.append('Also give a weight for each indicator showing how much it affected')
    lines.append('your score. The weights must add up to 1.')
    # The example reply uses placeholders, NOT real numbers. An earlier version
    # showed {"score": 50, ...} and the LLM copied it: most scores came out at
    # 50-55 whatever the data said (this is called "anchoring").
    lines.append('Reply ONLY as JSON in this format, with your own numbers in place of the <...>:')
    example_weights = ', '.join(f'"{col}": <weight>' for col in feature_cols)
    lines.append('{"score": <your score from 0 to 100>, "weights": {' + example_weights + '}}')
    return '\n'.join(lines)


def build_rephrased_prompt(country, row):
    lines = ['Here are economic statistics for ' + country + ':']
    for col in feature_cols:
        lines.append(f'- {col}: {row[col]:.2f}')
    lines.append('')
    lines.append('On a scale where 0 is the weakest possible economy and 100 is the')
    lines.append('strongest, how would you rate this economy?')
    lines.append('Reply ONLY as JSON: {"score": number}')
    return '\n'.join(lines)


def build_words_prompt(country, row):
    # The same numbers, written as sentences
    lines = [
        f'In {country}, prices rose by {row["inflation"]:.1f}% over the year.',
        f'{row["unemployment"]:.1f}% of people who want a job cannot find one.',
        f'Foreign companies invested an amount equal to {row["foreign_direct_investment"]:.1f}% of the economy.',
        f'The government owes debts equal to {row["government_debt"]:.1f}% of the economy.',
        f'Imports plus exports are equal to {row["trade_openness"]:.1f}% of the economy.',
        f'People live {row["human_development"]:.1f} years on average.',
        f'The government spends {row["education_spending"]:.1f}% of the economy on education.',
        '',
        'Give an overall economic health score from 0 (very poor) to 100 (excellent).',
        'Reply ONLY as JSON: {"score": number}'
    ]
    return '\n'.join(lines)


# ----- Talking to the LLM -----

def ask_llm(prompt):
    response = ollama.chat(
        model=llm_model,
        messages=[{'role': 'user', 'content': prompt}],
        format='json',                  # forces the reply to be valid JSON
        options={'temperature': 0}      # same question -> same answer
    )
    return response['message']['content']


def parse_score(text):
    try:
        data = json.loads(text.strip())
        score = float(data['score'])
        # Keep the score inside 0-100 in case the LLM goes outside it
        return max(0.0, min(100.0, score))
    except Exception:
        return None   # if anything goes wrong, we have no score


def parse_weights(text):
    try:
        data = json.loads(text.strip())
        weights = {}
        for col in feature_cols:
            weights[col] = abs(float(data['weights'][col]))
        # Make sure the weights add up to exactly 1
        total = sum(weights.values())
        for col in feature_cols:
            weights[col] = weights[col] / total
        return weights
    except Exception:
        return None


# ----- Scoring all countries -----

def score_with_llm(df):
    results = []

    for idx, row in df.iterrows():
        country = row['country_name']
        result = {'country': row['country']}

        # 1. Main prompt: score + weights
        text = ask_llm(build_prompt(country, row))
        result['llm_score'] = parse_score(text)
        weights = parse_weights(text)
        for col in feature_cols:
            if weights is None:
                result['weight_' + col] = None
            else:
                result['weight_' + col] = weights[col]

        # 2. Same numbers, question asked differently
        text = ask_llm(build_rephrased_prompt(country, row))
        result['llm_score_rephrased'] = parse_score(text)

        # 3. Same numbers, written as sentences
        text = ask_llm(build_words_prompt(country, row))
        result['llm_score_words'] = parse_score(text)

        print(f"{country}: main={result['llm_score']}  rephrased={result['llm_score_rephrased']}  words={result['llm_score_words']}")
        results.append(result)

    # Note: the guide suggests stretching the scores so the lowest becomes 0
    # and the highest 100. We do NOT do that, because it would hide exactly
    # what we want to measure: whether the LLM rates countries too high or
    # too low compared with the data.
    llm_df = pd.DataFrame(results)
    llm_df.to_csv('llm_scores.csv', index=False)
    print('Saved llm_scores.csv')
    return llm_df


if __name__ == '__main__':
    # Stop early with a clear message if Ollama is not running
    try:
        ollama.list()
    except Exception:
        print('Cannot reach Ollama. Install it from ollama.com, run "ollama pull mistral",')
        print('make sure the Ollama app is running, then try again.')
        raise SystemExit(1)

    df = pd.read_csv('economic_data.csv')
    latest = df[df['year'] == latest_year]
    score_with_llm(latest)
