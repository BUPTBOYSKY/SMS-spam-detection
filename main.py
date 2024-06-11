from data import SMSDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from model import BertBiLSTMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert labels to binary (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])


# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create train and test datasets
train_dataset = SMSDataset(train_df, tokenizer, max_length=128)
test_dataset = SMSDataset(test_df, tokenizer, max_length=128)

# Create train and test dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize the BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define hyperparameters
hidden_size = 128
num_layers = 2
dropout = 0.1

# Create an instance of the BertBiLSTMClassifier
model = BertBiLSTMClassifier(bert_model, hidden_size, num_layers, dropout)

# Set the device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model.to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}')
    
    # Evaluation
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.squeeze(), labels)
            
            test_loss += loss.item()
            preds = torch.sigmoid(logits.squeeze()).round()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
    print(f'Test Accuracy: {accuracy_score(test_labels, test_preds):.4f}')
    print(f'Test Precision: {precision_score(test_labels, test_preds):.4f}')
    print(f'Test Recall: {recall_score(test_labels, test_preds):.4f}')
    print(f'Test F1-score: {f1_score(test_labels, test_preds):.4f}')