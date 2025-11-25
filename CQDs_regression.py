import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
from sklearn import metrics
import numpy as np
import pandas as pd
import time
import random
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import datetime
import math
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
import pickle


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



root_dir = str(Path(os.getcwd()).parent)
from_dir =  'Data'  
to_dir =  'Results/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dummy_contrast_encode(df1, feature_list1):
    """ Dummy contrast encoding for categorical features. """
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


def load_data(task=0):
    """ Load CSV file into dataframes based on the task. """
    assert (task == 0 or task == 1 or task == 2), 'Error: invalid task spec'
    if task == 0:
        print('Loading MoS2 dataset...')
        df1 = pd.read_csv(from_dir + '/mos2_raw.csv')
        feature_list1 = df1.columns[0:(df1.shape[1] - 1)]
        df = dummy_contrast_encode(df1, feature_list1)
    elif task == 1:
        print('Loading CQD dataset...')
        df = pd.read_csv( from_dir+'/cqd_raw.csv')
    return df


def load_XY(task=0):
    """ Load dataset into X and Y arrays. """
    df = load_data(task)
    feature_list = df.columns[0:len(df.columns) - 1]
    result_col = df.columns[len(df.columns) - 1]

    X = df[feature_list]
    Y = df[result_col]
    return X, Y


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


def test(y_pred, y_true, end_ptr=None):
    """
    Evaluate the performance of predictions through several metrics:
    - R-squared (r2)
    - Mean Squared Error (mse)
    - Pearson correlation (pearsonr)
    """
    # All sample metrics
    r2 = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    pear, p_value = pearsonr(y_true, y_pred)

    # Print metrics
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Pearson Correlation: {pear:.4f}")

    return mse, rmse, r2, pear



X_df, Y_df = load_XY(task=1)  
X = X_df.values
Y = Y_df.values / 100 


scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# Save the scaler to a file
with open('Results/Regression_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
X_train = torch.tensor((X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0), dtype=torch.float32).to(device)
X_test = torch.tensor((X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0), dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)


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

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, seq_len=1, d_model=64, num_heads=16, ff_dim=512, num_layers=4, dropout=0.2, max_len=5000):
        super(TransformerRegressor, self).__init__()
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

        return x



def train_model_and_save_best(model, X_train, Y_train, X_test, Y_test, epochs=1000, batch_size=64, save_path="best_model.pth"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    
    best_loss = float('inf')

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    train_data = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()  
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)


        if epoch % 10 == 0:
            model.eval() 
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs.squeeze(), Y_test)
                print(f"Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss:.6f}, Test Loss: {test_loss.item():.6f}")
                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    torch.save(model.state_dict(), save_path)
                    print(f"Best model saved with Test Loss: {best_loss:.6f}")

            scheduler.step(test_loss.item())


input_dim = X_train.shape[1]  
seq_len = 1  
model = TransformerRegressor(input_dim=input_dim, d_model=128, num_heads=8, ff_dim=64, num_layers=1, dropout=0.1).to(device)
train_model_and_save_best(model, X_train, Y_train, X_test, Y_test, epochs=1000, batch_size=16, save_path="Results/HAT_Net_CQDs.pth")



#Evalavtion of the Traind model
model.load_state_dict(torch.load('Results/HAT_Net_CQDs.pth'))
model.to(device)
model.eval()
def plot_actual_vs_predicted_line(y_true, y_pred):
    plt.figure(figsize=(18, 10))
    plt.plot(y_true, label="Actual CQD Yield", color='blue', linestyle='-', linewidth=3)
    plt.plot(y_pred, label="Predicted CQD Yield", color='orange', linestyle='-', linewidth=3)
    plt.title("Actual vs Predicted CQD Yield ")
    plt.xlabel("Samples")
    plt.ylabel("CQD Yield")
    plt.legend(loc='upper right')
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'Results/actual_vs_predicted_line_{now_time}.png')
with torch.no_grad():
    Y_pred = model(X_test.clone().detach())

Y_pred_cpu = Y_pred.cpu().numpy()
Y_test_cpu = Y_test.cpu().numpy()

Y_pred_cpu = Y_pred_cpu.astype(np.float64).squeeze()  
Y_test_cpu = Y_test_cpu.astype(np.float64)
print(f"Y_pred_cpu type: {Y_pred_cpu.dtype}, shape: {Y_pred_cpu.shape}")
print(f"Y_test_cpu type: {Y_test_cpu.dtype}, shape: {Y_test_cpu.shape}")

if np.isnan(Y_pred_cpu).any() or np.isnan(Y_test_cpu).any():
    raise ValueError("NaNs found in predictions or test data!")

mse, rmse, r2, pear = test(Y_pred_cpu, Y_test_cpu)
Y_test_original = Y_test_cpu * 100
Y_pred_original = Y_pred_cpu * 100

print("=" * 50)
print("    \n Actual vs Predicted Photoluminescent Quantum Yield (PLQY):")

i = 0  # Initialize counter
print(f"{'Index':<10}{'Actual':<20}{'Predicted':<20}")
print("=" * 50)


# Evaluate the model
mse = metrics.mean_squared_error(Y_test_cpu, Y_pred_cpu)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(Y_test_cpu, Y_pred_cpu)
#pearson_corr, _ = metrics.pearsonr(Y_test_cpu, Y_pred_cpu)

# Print evaluation metrics
#print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Pearson: {pearson_corr:.4f}")


for predicted, actual in zip(Y_pred_original, Y_test_original):
    # Ensure values are scalars
    predicted = predicted.item() if isinstance(predicted, (np.ndarray, list)) else predicted
    actual = actual.item() if isinstance(actual, (np.ndarray, list)) else actual
    
    # Skip printing if actual value is 0.00 or predicted value is negative
    if actual != 0.00 and predicted >= 0.00:
        i += 1
        
        # Print header message after every 25 rows
        if (i - 1) % 21 == 0 and i > 1:
            print("=" * 50)
            print("\nActual vs Predicted Photoluminescent Quantum Yield (PLQY):")
            print(f"{'Index':<10}{'Actual':<20}{'Predicted':<20}")
            print("=" * 50)
        
        # Print the actual and predicted values
        print(f"{i:<10}{actual:<20.2f}{predicted:<20.2f}")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
plot_actual_vs_predicted_line(Y_test_original[20:81], Y_pred_original[30:75])
results_df = pd.DataFrame({'Actual': Y_test_original, 'Predicted': Y_pred_original})
save_csv(results_df, 'actual_vs_predicted_original_range', ind=False)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual', y='Predicted', data=results_df, color="dodgerblue", s=50, alpha=0.7, edgecolor="k")
plt.plot([0, 100], [0, 100], 'r--', linewidth=2)  # Perfect prediction line
plt.xlabel('Actual Quantum Yield (%)')
plt.ylabel('Predicted Quantum Yield (%)')
#plt.title('Actual vs Predicted Photo Luminance Quantum Yield')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Results/actual_vs_predicted_scatter.png')

# Density Plot for Predicted vs Actual
plt.figure(figsize=(10, 6))
sns.kdeplot(results_df['Actual'], label='Actual', fill=True, color="lightcoral", linewidth=2)
print(results_df['Actual'],'\n')
sns.kdeplot(results_df['Predicted'], label='Predicted', fill=True, color="lightblue", linewidth=2)
print(results_df['Predicted'],'\n')
plt.xlabel('Quantum Yield (%)')
plt.ylabel('Density')
#plt.title('Density Plot of Actual and Predicted Quantum Yield')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Results/actual_vs_predicted_density.png')

# Error Distribution Plot
errors = results_df['Predicted'] - results_df['Actual']
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, color="purple", bins=30)
plt.xlabel('Prediction Error (Predicted - Actual)')
plt.ylabel('Frequency')
plt.title('Error Distribution for Predicted Quantum Yield')
plt.grid(True)
plt.tight_layout()
plt.savefig('Results/error_distribution.png')