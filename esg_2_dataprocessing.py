#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare the ML training dataset by combining ESG report text summaries with
LSEG Environmental (E) scores.

Workflow:
    1. Load LSEG ESG scores (output of esg_0_get_esg_score_LSEG.py).
    2. For each company, load its text summary (output of esg_1_multi-files.py)
       and pair it with the corresponding E Score.
    3. Assign a score category label based on fixed thresholds.
    4. Plot the E Score distribution as a histogram.
    5. Save the final dataset to CSV for use in model training (esg_3_score.py).

Output columns: ticker, text, score, category

Score category thresholds:
    0 — E Score < 50  (low)
    1 — 50 <= E Score < 75  (medium)
    2 — E Score >= 75  (high)

Requirements:
    - pandas
    - matplotlib
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Settings
# =============================================================================
LSEG_SCORE_PATH  = 'LSEG_Score.csv'
SUMMARY_DIR      = './esg_summary/summary_df'
OUTPUT_FILE      = 'ESG_summary_and_LSEG_Score.csv'

# E Score bin boundaries and corresponding category labels
SCORE_BINS   = [-float('inf'), 50, 75, float('inf')]
SCORE_LABELS = [0, 1, 2]


# =============================================================================
# Functions
# =============================================================================
def load_lseg_scores(path):
    """
    Load LSEG ESG scores from CSV.

    Args:
        path (str): Path to the LSEG score CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns ['Ticker', 'ESG Score', 'E Score'].
    """
    df = pd.read_csv(path)
    df.columns = ['Ticker', 'ESG Score', 'E Score']
    return df


def find_matching_folders(summary_dir, ticker_list):
    """
    Find summary folders whose ticker matches a given list.
    Expects folder names in the format: <EXCHANGE>_<TICKER>_<YEAR>

    Args:
        summary_dir (str):        Path to the directory containing summary folders.
        ticker_list (pd.Series):  Series of valid ticker symbols.

    Returns:
        list: Folder names that match a ticker in ticker_list.
    """
    matched = []
    for folder in os.listdir(summary_dir):
        parts = folder.split('_')
        if len(parts) == 3 and parts[1] in ticker_list.values:
            matched.append(folder)
    return matched


def build_dataset(summary_dir, folder_list, df_score):
    """
    Build the training dataset by loading each company's text summary and
    pairing it with its E Score.

    Args:
        summary_dir (str):         Path to the directory containing summary folders.
        folder_list (list):        List of matched folder names.
        df_score    (pd.DataFrame): LSEG score DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns ['ticker', 'text', 'score'].
    """
    tickers, texts, scores = [], [], []

    for folder in folder_list:
        csv_path = os.path.join(summary_dir, folder, 'summary_context.csv')
        ticker   = folder.split('_')[1]

        try:
            data  = pd.read_csv(csv_path)
            text  = '\n'.join(f'{i+1}. {row["point"]}' for i, row in data.iterrows())
            score = df_score[df_score['Ticker'] == ticker]['E Score'].values[0]

            tickers.append(ticker)
            texts.append(text)
            scores.append(score)

        except Exception as e:
            print(f'  [SKIP] {folder}: {e}')

    return pd.DataFrame({'ticker': tickers, 'text': texts, 'score': scores})


def plot_score_distribution(scores, bins, output_path=None):
    """
    Plot a histogram of E Score distribution with category boundary lines.

    Args:
        scores      (pd.Series): E Score values.
        bins        (list):      Score bin boundaries (excluding inf values).
        output_path (str):       If provided, saves the figure to this path.
                                 Otherwise, displays it interactively.
    """
    thresholds = [b for b in bins if b not in (-float('inf'), float('inf'))]

    plt.figure()
    plt.hist(scores, bins=10, edgecolor='black')
    for threshold in thresholds:
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                    label=f'threshold = {threshold}')
    plt.title('E Score Distribution')
    plt.xlabel('E Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f'  Plot saved to {output_path}')
    else:
        plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    # --- Load LSEG scores ---
    df_score    = load_lseg_scores(LSEG_SCORE_PATH)
    ticker_list = df_score['Ticker']
    print(f'Loaded {len(ticker_list)} tickers from {LSEG_SCORE_PATH}')

    # --- Match summary folders to tickers ---
    folder_list = find_matching_folders(SUMMARY_DIR, ticker_list)
    print(f'Matched {len(folder_list)} summary folders')

    # --- Build dataset ---
    data = build_dataset(SUMMARY_DIR, folder_list, df_score)
    data = data.dropna()
    print(f'Dataset size after dropping NaN: {len(data)}')

    # --- Assign score categories ---
    data['category'] = pd.cut(data['score'], bins=SCORE_BINS, labels=SCORE_LABELS)
    print(data['category'].value_counts().sort_index().rename('count'))

    # --- Plot E Score distribution ---
    plot_score_distribution(data['score'], SCORE_BINS)

    # --- Save ---
    data.to_csv(OUTPUT_FILE, index=False)
    print(f'\nSaved {len(data)} records to {OUTPUT_FILE}')
