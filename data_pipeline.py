# data_pipeline.py  (Part 2 of the NationPulse guide)
#
# Downloads economic data for 52 countries from the World Bank and the IMF,
# fills small gaps, and saves it as economic_data.csv
# (one row per country per year).
#
# Data sources (see CREDITS.md for licences and how to cite them):
#   - World Bank, World Development Indicators (CC BY 4.0), which itself draws
#     on the IMF, ILO, UN Population Division, UNESCO Institute for Statistics,
#     OECD and national statistical offices
#   - IMF, World Economic Outlook database, through the IMF DataMapper (free
#     reuse with credit to the IMF; commercial use needs the IMF's permission)
#
# Run:  python data_pipeline.py

import requests      # used to download data from the internet
import pandas as pd  # used to work with tables of data


# ----- Settings: what data we want -----

# Each World Bank indicator code, and the column name we want to give it
indicators = {
    'NY.GDP.PCAP.CD': 'gdp_per_capita',
    'FP.CPI.TOTL.ZG': 'inflation',
    'SL.UEM.TOTL.ZS': 'unemployment',
    'BX.KLT.DINV.WD.GD.ZS': 'foreign_direct_investment',
    'NE.TRD.GNFS.ZS': 'trade_openness',
    'SP.DYN.LE00.IN': 'human_development',   # life expectancy
    'SE.XPD.TOTL.GD.ZS': 'education_spending'
}

# Government debt comes from the IMF instead of the World Bank, because the
# World Bank has no debt data at all for about half of our countries.
# GGXWDG_NGDP = "General government gross debt, percent of GDP"
imf_debt_code = 'GGXWDG_NGDP'

# Two-letter codes of the 52 countries, grouped by region.
# They cover every region and every income level (low, middle, high).
countries = ['IN', 'PK', 'BD', 'LK', 'NP',                         # South Asia
             'CN', 'JP', 'KR', 'ID', 'TH', 'VN', 'MY', 'PH',       # East Asia & Pacific
             'AU', 'NZ',
             'NG', 'ZA', 'KE', 'ET', 'GH', 'TZ',                   # Sub-Saharan Africa
             'UG', 'RW', 'SN', 'CI', 'MZ',
             'BR', 'MX', 'AR', 'CO', 'CL', 'PE', 'EC', 'BO',       # Latin America
             'DE', 'FR', 'GB', 'IT', 'ES', 'PL', 'NL',             # Europe & Central Asia
             'UA', 'KZ', 'UZ',
             'EG', 'MA', 'SA', 'TR', 'IR', 'JO',                   # Middle East & North Africa
             'US', 'CA']                                           # North America

# The years we want
start_year = 2014
end_year = 2023


# ----- Step 1: Download World Bank indicators -----

def fetch_world_bank(country_codes, rows):
    # The World Bank wants the countries joined by ';', like "IN;PK;BD"
    country_text = ';'.join(country_codes)

    for code, column_name in indicators.items():
        print('Downloading', column_name, '...')

        url = 'https://api.worldbank.org/v2/country/' + country_text + '/indicator/' + code
        settings = {
            'format': 'json',                            # we want the answer as JSON
            'date': str(start_year) + ':' + str(end_year),
            'per_page': 20000                            # get everything in one page
        }
        response = requests.get(url, params=settings, timeout=60)
        answer = response.json()

        # The answer is a list of two parts:
        #   answer[0] = information about the page
        #   answer[1] = the actual data (one record per country per year)
        if len(answer) < 2 or answer[1] is None:
            print('  World Bank sent no data for', code, '- skipping it')
            continue
        records = answer[1]

        for record in records:
            country = record['countryiso3code']   # e.g. 'IND'
            year = int(record['date'])            # e.g. 2020
            value = record['value']               # the number (None if missing)

            # rows is a dictionary with one entry per (country, year).
            # Example: rows[('IND', 2020)] = {'country': 'IND', 'year': 2020, 'inflation': 6.6, ...}
            key = (country, year)
            if key not in rows:
                rows[key] = {'country': country, 'year': year}
            rows[key][column_name] = value


# ----- Step 2: Download government debt from the IMF -----

def fetch_imf_debt(rows):
    print('Downloading government_debt (IMF) ...')

    url = 'https://www.imf.org/external/datamapper/api/v1/' + imf_debt_code
    response = requests.get(url, timeout=60)
    answer = response.json()

    # The IMF sends every country and every year, organised like this:
    #   answer['values']['GGXWDG_NGDP']['IND']['2020'] = 90.4
    imf_debt = answer['values'][imf_debt_code]

    # The IMF uses three-letter country codes (like 'IND'), the same codes our
    # rows already use, so we can add debt to each existing row directly.
    for key in rows:
        country = rows[key]['country']
        year = rows[key]['year']
        debt_by_year = imf_debt.get(country, {})                     # {} if country not found
        rows[key]['government_debt'] = debt_by_year.get(str(year))  # None if year not found


# ----- Step 3: Download each country's name, region and income level -----
# Later parts need these: the LLM needs the country's name, and the
# comparison checks each system's errors by region and income level.

# The World Bank uses official names like "Egypt, Arab Rep.". LLMs know
# countries by their everyday names, so we replace these ones.
common_names = {
    'EGY': 'Egypt',
    'IRN': 'Iran',
    'KOR': 'South Korea',
    'VNM': 'Vietnam',
    'CIV': 'Ivory Coast',
    'TUR': 'Turkey'
}

def fetch_country_info(country_codes):
    print('Downloading country names, regions and income levels ...')

    url = 'https://api.worldbank.org/v2/country/' + ';'.join(country_codes)
    response = requests.get(url, params={'format': 'json', 'per_page': 500}, timeout=60)
    answer = response.json()

    info_rows = []
    for record in answer[1]:
        code = record['id']                                           # e.g. 'IND'
        name = common_names.get(code, record['name'])                 # everyday name if we have one
        info_rows.append({
            'country': code,
            'country_name': name,                                     # e.g. 'India'
            'region': record['region']['value'].strip(),              # e.g. 'South Asia'
            'income_level': record['incomeLevel']['value'].strip()    # e.g. 'Lower middle income'
        })
    return pd.DataFrame(info_rows)


# ----- Step 4: Fill in missing values -----
#
# These indicators change slowly from one year to the next, so the best guess
# for a missing year is based on the SAME country's nearby years:
#
#   - A gap between two known years is filled with a value in between
#     (e.g. 2017 = 4.0, 2018 = missing, 2019 = 6.0  ->  2018 = 5.0).
#   - A gap at the start or end is filled with the nearest known year.
#
# We never copy values from other countries, because countries are very
# different (Japan's debt is not a good guess for Kenya's). So if a country
# has no data at all for an indicator, it stays empty (NaN), which honestly
# says "unknown" instead of making up a number.

def fill_gaps(values):
    values = values.interpolate()   # fill gaps between known years
    values = values.ffill()         # fill gaps at the end with the last known year
    values = values.bfill()         # fill gaps at the start with the first known year
    return values


# ----- Put it all together -----

def build_dataset(country_codes):
    rows = {}
    fetch_world_bank(country_codes, rows)
    fetch_imf_debt(rows)

    df = pd.DataFrame(list(rows.values()))
    df = df.sort_values(['country', 'year'])
    df = df.reset_index(drop=True)

    print('Missing values before filling:')
    print(df.isna().sum())

    columns_to_fill = list(indicators.values()) + ['government_debt']
    for column_name in columns_to_fill:
        # Fill each country separately, so countries never mix
        df[column_name] = df.groupby('country')[column_name].transform(fill_gaps)

    print('Missing values after filling (countries with no data at all):')
    print(df.isna().sum())

    # Add name, region and income level to every row, matching on 'country'
    info = fetch_country_info(country_codes)
    df = pd.merge(info, df, on='country', how='inner')
    return df


# This block runs only when you start this file directly
# (python data_pipeline.py), not when another file imports from it.
if __name__ == '__main__':
    df = build_dataset(countries)
    df.to_csv('economic_data.csv', index=False)

    print(df.head())
    print('Countries:', df['country'].nunique(), '| Rows and columns:', df.shape)
    print('Saved economic_data.csv')