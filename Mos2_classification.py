import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import pickle
import seaborn as sns


root_dir = str(Path(os.getcwd()).parent)
from_dir = 'Data'
to_dir = 'Results/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Functions to save data and format the file names
def save_csv(data, title, ind=False):
    """ Save the results to a CSV file. """
    to_save_title = format_title(to_dir, title, fileEtd='.csv')
    data.to_csv(to_save_title, index=ind)
    print('Successfully saved:', to_save_title)
    return to_save_title

def update_title_w_date(title):
    """ Update title with the current date. """
    now_time = datetime.datetime.now()
    today = str(now_time.year) + '_' + str(now_time.month) + '_' + str(now_time.day)
    return title + today

def format_title(to_dir, title, fileEtd):
    """ Format the title for saving with unique file name. """
    title = update_title_w_date(title)
    to_save_title = to_dir + title + fileEtd
    i = 0
    while os.path.exists(to_save_title):
        to_save_title = to_dir + title + '_' + str(i) + fileEtd
        i = i + 1
    return to_save_title

# Contrast encoding for categorical features
def contrast_encode(df1, feature_list1):
    """  contrast encoding for categorical features. """
    new_df = pd.DataFrame()
    feature_dummies = {}
    for col in feature_list1:
        if str(df1[col].dtypes) == 'object':
            # handling categorical data
            temp = pd.get_dummies(df1[col], columns=[col], prefix=col)
            del temp[temp.columns[0]]
            feature_dummies[col] = list(temp.columns.values)
            for new_col in temp.columns:
                new_df[new_col] = temp[new_col]
        else:
            new_df[col] = df1[col]

    if str(df1[df1.columns[len(df1.columns) - 1]].dtypes) == 'object':
        le = LabelEncoder()
        le.fit(['Cannot', 'Can'])
        temp = 1 - le.transform(df1[df1.columns[len(df1.columns) - 1]])
        new_df['Result'] = temp
    else:
        new_df['Result'] = df1['Result']
    return new_df

# Load data function
def load_data(task=0):
    """ Load CSV file into dataframes based on the task. """
    assert (task == 0 or task == 1 or task == 2), 'Error: invalid task spec'
    if task == 0:
        print('Loading MoS2 dataset...')
        df1 = pd.read_csv(from_dir + '/mos2_raw.csv')
        feature_list1 = df1.columns[0:(df1.shape[1] - 1)]
        df = contrast_encode(df1, feature_list1)
    elif task == 1:
        print('Loading CQD dataset...')
        df = pd.read_csv(from_dir + '/cqd_raw.csv')
    return df

# Load X and Y arrays from dataset
def load_XY(task=0):
    df = load_data(task)
    feature_list = df.columns[0:len(df.columns) - 1]
    result_col = df.columns[len(df.columns) - 1]
    X = df[feature_list]
    Y = df[result_col]
    return X, Y

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        # Learnable positional encoding
        self.position_embeddings = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        position_encodings = self.position_embeddings(positions)
        return x + position_encodings

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len=1, d_model=64, num_heads=16, ff_dim=512, num_layers=4, dropout=0.2, max_len=5000):
        super(TransformerClassifier, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection to match d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Learnable Positional Encoding
        self.positional_encoding = LearnablePositionalEncoding(d_model, max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layers with increased capacity
        self.fc1 = nn.Linear(d_model * seq_len, 256)  # Increased size
        self.layer_norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.layer_norm2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        # Dropout and regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm3 = nn.LayerNorm(64)

    def forward(self, x):
        # Project input to d_model
        x = self.input_projection(x)  # Shape: (batch_size, seq_len=1, d_model)

        # Add sequence dimension (seq_len should be > 1 for transformers)
        x = x.unsqueeze(1)  # Shape: (batch_size, seq_len=1, d_model)

        # Apply learnable positional encoding
        x = self.positional_encoding(x)  # Shape: (batch_size, seq_len=1, d_model)

        # Transpose to match transformer input shape
        x = x.transpose(0, 1)  # Shape: (seq_len=1, batch_size, d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)  # Shape: (seq_len, batch_size, d_model)

        # Pool across sequence dimension
        x = x.mean(dim=0)  # Shape: (batch_size, d_model)

        # Fully connected layers with normalization and dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.layer_norm1(x)
        
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.layer_norm2(x)

        x = torch.relu(self.fc3(x))
        x = self.layer_norm3(x)
        x = self.fc4(x)

        # Apply sigmoid for binary classification
        return torch.sigmoid(x).squeeze()


# Function to train model and save the best one
def train_model_and_save_best(model, X_train, Y_train, X_test, Y_test, epochs=2000, batch_size=120, save_path="best_model_classification.pth"):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_loss = float('inf')

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    train_data = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, Y_test)

                print(f"Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss:.6f}, Test Loss: {test_loss.item():.6f}")

                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    torch.save(model.state_dict(), save_path)
                    print(f"Best model saved with Test Loss: {best_loss:.6f}")

            scheduler.step(test_loss.item())

# Load and process the data
X_df, Y_df = load_XY(task=0)  
X = X_df.values
Y = Y_df.values

# Split original data (before scaling)
X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(X, Y, test_size=0.1, random_state=42)

# Combine X and Y data into a single DataFrame for both train and test sets
train_data_orig = pd.DataFrame(X_train_orig, columns=X_df.columns)
train_data_orig['Result'] = Y_train_orig
test_data_orig = pd.DataFrame(X_test_orig, columns=X_df.columns)
test_data_orig['Result'] = Y_test_orig

# Save combined original train/test data to CSV files
save_csv(train_data_orig, 'train_data_original', ind=False)
save_csv(test_data_orig, 'test_data_original', ind=False)

# Scale the data
# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler to a file
with open('Results/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split scaled data for model training
X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Convert scaled data to tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

# Initialize model and train
input_dim = X_train.shape[1]
seq_len = 1  
model = TransformerClassifier(input_dim=input_dim, seq_len=seq_len, d_model=128, num_heads=2, ff_dim=122, num_layers=8, dropout=0.1).to(device)
train_model_and_save_best(model, X_train, Y_train, X_test, Y_test, epochs=1000, batch_size=32, save_path="Results/_Refined_Mos2_HAT_Net.pth")




