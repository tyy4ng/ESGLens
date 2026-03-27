#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch LSEG (Refinitiv) ESG scores for a list of stock tickers and save the
results to a CSV file.

Workflow:
    1. Load a list of tickers (from a pre-defined index or the local ESG report folder).
    2. Fetch ESG / Environmental / Social / Governance pillar scores via the
       Refinitiv Data (rd) API, trying multiple RIC suffixes (.O, plain, .K).
    3. Pivot the raw API responses into a single wide-format DataFrame.
    4. Save the final DataFrame to CSV.

Output columns: ticker, ESG Score, E Score

Requirements:
    - refinitiv.data
    - pandas
"""

import os
import refinitiv.data as rd
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)


# =============================================================================
# Settings
# =============================================================================
TARGET_YEAR     = 2022
OUTPUT_FILENAME = 'LSEG_Score.csv'
REPORT_DIR      = './esg_report'

FIELDS = [
    'TR.TRESGScore',
    'TR.EnvironmentPillarScore',
    'TR.SocialPillarScore',
    'TR.GovernancePillarScore',
]

# API parameters: annual data, fetch up to 24 fiscal years back from today
API_PARAMS = {'Period': 'FY0', 'Frq': 'FY', 'SDate': 0, 'EDate': -24}


# =============================================================================
# Ticker loaders
# =============================================================================
def get_QQQ_tickers():
    """Load ticker symbols from a QQQ holdings CSV file."""
    df = pd.read_csv('./esg_tickers/QQQ-holdings.csv')
    return df[df['Symbol'] != '--']['Symbol'].tolist()


def get_SP500_tickers():
    """Load ticker symbols from an S&P 500 CSV file."""
    df = pd.read_csv('./esg_tickers/SP500.csv')
    return df['Ticker'].tolist()


def get_Russell1000_tickers():
    """Load ticker symbols from a Russell 1000 CSV file."""
    df = pd.read_csv('./esg_tickers/Russell1000.csv')
    return df['Ticker'].tolist()


def get_tickers_from_reports(report_dir):
    """
    Derive ticker symbols from ESG report filenames.
    Expects filenames in the format: <EXCHANGE>_<TICKER>_<YEAR>.pdf
    Example: NYSE_AAPL_2022.pdf -> AAPL
    """
    filenames = os.listdir(report_dir)
    return [f.split('_')[1] for f in filenames if f.endswith('.pdf')]


# =============================================================================
# Data fetching
# =============================================================================
def fetch_data(ticker, df_list, failed_list):
    """
    Fetch ESG scores for a single ticker from the Refinitiv API.

    Tries three RIC suffixes in order: '.O', '', '.K'.
    On success, appends the raw DataFrame to df_list.
    On failure for all suffixes, appends the ticker to failed_list.

    Args:
        ticker      (str):  Base ticker symbol (e.g. 'AAPL').
        df_list     (list): Accumulator for successful API responses.
        failed_list (list): Accumulator for tickers that could not be fetched.
    """
    for suffix in ['.O', '', '.K']:
        ticker_ric = ticker + suffix
        try:
            df = rd.get_data(
                universe=[ticker_ric],
                fields=FIELDS,
                parameters=API_PARAMS,
            )
            df_list.append(df)
            print(f"  [OK] {ticker_ric}")
            return
        except Exception as e:
            print(f"  [FAIL] {ticker_ric}: {e}")

    print(f"  [SKIP] {ticker} — all suffixes failed")
    failed_list.append(ticker)


# =============================================================================
# Data processing
# =============================================================================
def pivot_yearly(df, score_column):
    """
    Reshape a raw Refinitiv API response into a single-column yearly DataFrame.

    Args:
        df           (pd.DataFrame): Raw response from rd.get_data().
        score_column (str):          Column name to extract (e.g. 'ESG Score').

    Returns:
        pd.DataFrame: Two-column DataFrame with 'Year' and the ticker as columns.
    """
    ticker = df['Instrument'].iloc[0].split('.')[0]  # strip RIC suffix
    df = df[['Date', score_column]].copy()
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df = df.groupby('Year').last().reset_index()
    df.drop(columns=['Date'], inplace=True)
    df.columns = ['Year', ticker]
    return df


def merge_scores(df_list, score_column, target_year):
    """
    Merge per-ticker yearly DataFrames and extract a single target year as a
    ticker-indexed Series.

    Args:
        df_list      (list): List of raw DataFrames from rd.get_data().
        score_column (str):  Column name to extract.
        target_year  (int):  The fiscal year to keep (e.g. 2022).

    Returns:
        pd.DataFrame: Single-row DataFrame filtered to target_year, transposed
                      so each row is one ticker.
    """
    pivoted = [pivot_yearly(df, score_column) for df in df_list]
    merged = pivoted[0]
    for p in pivoted[1:]:
        merged = pd.merge(merged, p, on='Year', how='outer')
    return merged[merged['Year'] == target_year].iloc[:, 1:].T


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    # --- Load tickers ---
    # Uncomment the index you want to use, or use get_tickers_from_reports()
    # tickers = get_QQQ_tickers()
    # tickers = get_SP500_tickers()
    # tickers = get_Russell1000_tickers()
    tickers = get_tickers_from_reports(REPORT_DIR)
    print(f"Total tickers to fetch: {len(tickers)}")

    # --- Fetch from Refinitiv ---
    rd.open_session()

    df_list     = []
    failed_list = []
    for ticker in tickers:
        fetch_data(ticker, df_list, failed_list)

    rd.close_session()

    if failed_list:
        print(f"\nFailed tickers ({len(failed_list)}): {failed_list}")

    # --- Build output DataFrame ---
    esg_scores = merge_scores(df_list, 'ESG Score',                  TARGET_YEAR)
    e_scores   = merge_scores(df_list, 'Environmental Pillar Score', TARGET_YEAR)

    result = pd.merge(esg_scores, e_scores, left_index=True, right_index=True)
    result.reset_index(inplace=True)
    result.columns = ['ticker', 'ESG Score', 'E Score']

    # --- Save ---
    result.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\nSaved {len(result)} records to '{OUTPUT_FILENAME}'")
