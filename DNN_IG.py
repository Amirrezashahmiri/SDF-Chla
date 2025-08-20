import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import os
import sys

# Parameters
file_path = r"C:\Users\DFMRendering\Desktop\Oman climate week\Visualization\Maybe_Final\Issue\output_real_data.xlsx"
output_dir = os.path.dirname(file_path) 

# Debug Mode Parameters
debug_mode = False
debug_fraction = 1


# Number of K-Fold splits
n_splits = 10

# Load data
try:
    data = pd.read_excel(file_path)
    print(f"Data loaded successfully with shape: {data.shape}")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    sys.exit(1)

# If debug_mode is enabled, sample a subset of the data
if debug_mode:
    if not (0 < debug_fraction <= 1):
        print("debug_fraction must be between 0 and 1.")
        sys.exit(1)
    data = data.sample(frac=debug_fraction, random_state=42).reset_index(drop=True)
    print(f"Running in debug mode with {len(data)} rows.")

# Exclude specific columns from modeling but retain them for reporting
columns_to_exclude = ['year_month', 'Latitude', 'Longitude', 'date', 'data_type', '.geo']
excluded_data = data[columns_to_exclude]
modeling_data = data.drop(columns=columns_to_exclude)

# Define features and target
target_col = 'Next month Chl-a (mg/m³)'
if target_col not in modeling_data.columns:
    raise ValueError(f"Target column '{target_col}' not found in the data.")
features = modeling_data.drop(columns=[target_col])
target = modeling_data[target_col].values

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Define the neural network model
class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation_function):
        super(Net, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_function)
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Function to perform K-fold cross-validation with fixed hyperparameters
def cross_validate_and_interpret(model_class, X_data, y_data, n_splits, epochs, batch_size, hidden_sizes, activation_function, optimizer_name):
    if len(X_data) < n_splits:
        raise ValueError(f"Number of samples ({len(X_data)}) is less than the number of folds ({n_splits}).")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_attributions_all_folds = []
    
    for fold_num, (train_index, test_index) in enumerate(kf.split(X_data), 1):
        print(f"Starting fold {fold_num}/{n_splits}")
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
        
        # Initialize a new model instance for each fold
        model_fold = model_class(input_size=X_data.shape[1], hidden_sizes=hidden_sizes, activation_function=activation_function).to(device)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(model_fold.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Train the model with Early Stopping
        model_fold.train()
        best_loss = np.inf
        epochs_no_improve = 0
        early_stopping_patience = 20
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model_fold(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_epoch_loss)
            print(f"Fold {fold_num}, Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            # Check for improvement
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} for fold {fold_num}")
                break

        # Interpretation with Captum
        print(f"Interpreting fold {fold_num}")
        model_fold.eval()
        ig = IntegratedGradients(model_fold)
        attributions, delta = ig.attribute(X_test_tensor, target=0, return_convergence_delta=True)
        attributions = attributions.cpu().detach().numpy()
        
        # Compute average attributions for this fold
        avg_attributions = np.mean(attributions, axis=0)
        avg_attributions_all_folds.append(avg_attributions)

    # Compute average attributions across all folds
    avg_attributions_final = np.mean(avg_attributions_all_folds, axis=0)
    feature_names = features.columns

    # Print average attributions
    print("\nAverage Attributions Across All Folds:")
    for idx, feature in enumerate(feature_names):
        print(f"Feature: {feature}, Average Attribution: {avg_attributions_final[idx]:.4f}")

    # Plot feature importance
    plt.figure(figsize=(15, 9))
    bars = plt.barh(feature_names, avg_attributions_final, color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(feature_names))), edgecolor='black')
    plt.xlabel('Average Attribution', fontsize=18, fontweight='bold', family='Helvetica', labelpad=10)
    plt.xticks(fontsize=20, family='Helvetica')
    plt.ylabel('Features', fontsize=18, fontweight='bold', family='Helvetica', labelpad=10)
    plt.yticks(fontsize=16, family='Helvetica')
    plt.title('Feature Importance Based on Integrated Gradients', fontsize=20, fontweight='bold', family='Helvetica', pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Add value labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', 
                 va='center', fontsize=16, family='Helvetica')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=500)
    plt.show()
    print(f"Feature importance plot saved to {plot_path}")

# Define hyperparameters
hidden_sizes = [256, 128, 64]
activation_name = nn.ReLU()
batch_size = 30
epochs = 200

# Perform cross-validation with interpretation
try:
    cross_validate_and_interpret(
        Net, X_scaled, target, n_splits=n_splits,
        epochs=epochs, batch_size=batch_size,
        hidden_sizes=hidden_sizes,
        activation_function=activation_name,
        optimizer_name='Adam'
    )
except ValueError as ve:
    print(f"An error occurred: {ve}")
    sys.exit(1)