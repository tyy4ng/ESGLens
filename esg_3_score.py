#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune a BERT-based regression model to predict LSEG Environmental (E) scores
from ESG report text summaries.

Workflow:
    1. Load the training dataset (output of esg_2_dataprocessing.py).
    2. Tokenize the text and split into train / test sets.
    3. Fine-tune a regression head on top of a frozen BERT encoder.
    4. Evaluate predictions on the test set.
    5. Save the trained model.

Model architecture:
    - Encoder : bert-base-uncased (weights frozen during training)
    - Regressor: single Linear layer (hidden_size -> 1)

Alternative pre-trained model (domain-specific, commented out):
    - ESGBERT/EnvironmentalBERT-environmental

Requirements:
    - torch
    - transformers
    - scikit-learn
    - pandas
"""

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


# =============================================================================
# Settings
# =============================================================================
FILE_PATH     = 'ESG_summary_and_LSEG_Score.csv'   # output of esg_2_dataprocessing.py
MODEL_NAME    = 'bert-base-uncased'
OUTPUT_PATH   = 'esg_bert_regressor.pt'

LEARNING_RATE = 1e-4
BATCH_SIZE    = 2
EPOCHS        = 5
TEST_SIZE     = 0.2
RANDOM_STATE  = 2024


# =============================================================================
# Model
# =============================================================================
class BERTRegressionModel(nn.Module):
    """
    BERT encoder with a single regression head.

    The BERT encoder weights are frozen; only the regression layer is trained.
    The [CLS] token representation is used as the sentence embedding.

    Args:
        bert_model (BertModel): Pre-loaded BERT model instance.
    """

    def __init__(self, bert_model):
        super().__init__()
        self.bert       = bert_model
        self.regressor  = nn.Linear(bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        return self.regressor(cls_output)


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    # --- Load data ---
    df     = pd.read_csv(FILE_PATH)
    texts  = df['text'].tolist()
    labels = df['score'].tolist()
    print(f'Loaded {len(texts)} samples from {FILE_PATH}')

    # --- Load tokenizer and pre-trained model ---
    tokenizer        = BertTokenizer.from_pretrained(MODEL_NAME)
    pretrained_model = BertModel.from_pretrained(MODEL_NAME)

    # Alternative: domain-specific ESG model
    # from transformers import AutoTokenizer, AutoModel
    # tokenizer        = AutoTokenizer.from_pretrained('ESGBERT/EnvironmentalBERT-environmental')
    # pretrained_model = AutoModel.from_pretrained('ESGBERT/EnvironmentalBERT-environmental')

    # --- Train / test split ---
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f'Train: {len(texts_train)} | Test: {len(texts_test)}')

    # --- Tokenize ---
    inputs_train = tokenizer(texts_train, padding=True, truncation=True, return_tensors='pt')
    inputs_test  = tokenizer(texts_test,  padding=True, truncation=True, return_tensors='pt')

    labels_train = torch.tensor(labels_train, dtype=torch.float).unsqueeze(1)
    labels_test  = torch.tensor(labels_test,  dtype=torch.float).unsqueeze(1)

    # --- DataLoaders ---
    train_loader = DataLoader(
        TensorDataset(inputs_train['input_ids'], inputs_train['attention_mask'], labels_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(inputs_test['input_ids'], inputs_test['attention_mask'], labels_test),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # --- Build model ---
    model     = BERTRegressionModel(pretrained_model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=LEARNING_RATE)

    # --- Training ---
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for input_ids, attention_mask, target_scores in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs, target_scores)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss / len(train_loader):.4f}')

    # --- Save model ---
    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f'\nModel saved to {OUTPUT_PATH}')

    # --- Evaluation ---
    model.eval()
    print('\n--- Test Set Predictions ---')
    with torch.no_grad():
        for input_ids, attention_mask, target_scores in test_loader:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted = outputs.squeeze().tolist()
            true      = target_scores.squeeze().tolist()
            # handle single-sample batches (squeeze turns scalar into 0-d tensor)
            if not isinstance(predicted, list):
                predicted, true = [predicted], [true]
            for p, t in zip(predicted, true):
                print(f'  Predicted: {p:.2f}  |  True: {t:.2f}')
