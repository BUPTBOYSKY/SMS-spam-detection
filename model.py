import torch
import torch.nn as nn
from transformers import BertModel

class BertBiLSTMClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size, num_layers, dropout):
        super(BertBiLSTMClassifier, self).__init__()
        
        self.bert = bert_model
        self.bi_lstm = nn.LSTM(
            input_size=bert_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, input_ids, attention_mask):
        # Pass input through BERT
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract the last hidden state of the [CLS] token
        last_hidden_state = bert_output.last_hidden_state
        cls_hidden_state = last_hidden_state[:, 0, :]
        
        # Pass the [CLS] token representation through the BiLSTM
        lstm_output, _ = self.bi_lstm(cls_hidden_state.unsqueeze(1))
        
        # Apply dropout
        lstm_output = self.dropout(lstm_output)
        
        # Pass the BiLSTM output through the fully connected layer
        logits = self.fc(lstm_output.squeeze(1))
        
        return logits