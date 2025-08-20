import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import os
import math
import matplotlib.gridspec as gridspec

# Set global plot appearance for journal standards
plt.rcParams.update({
    'font.family': 'helvetica',
    'font.size': 14,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'figure.figsize': (10, 8),
    'savefig.dpi': 500
})

# Parameters
test_mode = True  # Set to True for testing with a smaller subset of the data
test_percentage = 1  # Percentage of data to use in test mode
file_path = "C:/Users/DFMRendering/Desktop/Oman climate week/Visualization/Maybe_Final/output_augmented_balanced.xlsx"

# Load data
try:
    data = pd.read_excel(file_path)
    print(f"Data loaded successfully with shape: {data.shape}")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit(1)

# Reduce dataset size for debugging if test_mode is True
if test_mode:
    data = data.sample(frac=test_percentage, random_state=42).reset_index(drop=True)
    print(f"Running in test mode with {len(data)} rows.")

# Exclude specific columns from modeling but retain them for reporting
columns_to_exclude = ['year_month', 'Latitude', 'Longitude', 'date', 'data_type', '.geo']
excluded_data = data[columns_to_exclude]
modeling_data = data.drop(columns=columns_to_exclude)

# Drop rows with missing values
initial_rows = len(modeling_data)
modeling_data = modeling_data.dropna()
excluded_data = excluded_data.loc[modeling_data.index]  # Align excluded data with remaining rows
removed_rows = initial_rows - len(modeling_data)
print(f"Number of rows removed due to missing values: {removed_rows}")

# Define features and target
target_col = 'Next month Chl-a (mg/m³)'
if target_col not in modeling_data.columns:
    raise ValueError(f"Target column '{target_col}' not found in the data.")
features = modeling_data.drop(columns=[target_col])
target = modeling_data[target_col]

# Initialize model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Set up cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store metrics and results
mae_scores, mse_scores, rmse_scores, r2_scores = [], [], [], []
predictions, true_values = [], []
rows_indices = []  # To track indices of rows used
feature_importances_list = []  # List to store feature importances for each fold

# Perform cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(features), 1):
    X_train, X_val = features.iloc[train_index], features.iloc[val_index]
    y_train, y_val = target.iloc[train_index], target.iloc[val_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Append metrics
    mae_scores.append(mean_absolute_error(y_val, y_pred))
    mse = mean_squared_error(y_val, y_pred)
    mse_scores.append(mse)
    rmse_scores.append(np.sqrt(mse))
    r2_scores.append(r2_score(y_val, y_pred))

    # Store feature importances
    feature_importances_list.append(model.feature_importances_)

    # Store predictions, true values, and row indices
    predictions.extend(y_pred)
    true_values.extend(y_val)
    rows_indices.extend(y_val.index)

    print(f"Fold {fold}: MAE={mae_scores[-1]:.2f}, MSE={mse_scores[-1]:.2f}, RMSE={rmse_scores[-1]:.2f}, R²={r2_scores[-1]:.2f}")

# Calculate average metrics and standard deviations
avg_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)
avg_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
avg_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
avg_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

print(f"\nAverage MAE: {avg_mae:.2f} ± {std_mae:.2f}")
print(f"Average MSE: {avg_mse:.2f} ± {std_mse:.2f}")
print(f"Average RMSE: {avg_rmse:.2f} ± {std_rmse:.2f}")
print(f"Average R²: {avg_r2:.2f} ± {std_r2:.2f}")

# Calculate average feature importances
avg_feature_importances = np.mean(feature_importances_list, axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': avg_feature_importances
}).sort_values(by='Importance', ascending=False)

# Combine predictions and true values with excluded columns
results_df = pd.DataFrame({
    'True': true_values,
    'Predicted': predictions
}, index=rows_indices)

# Add excluded columns
results_with_metadata = pd.concat([excluded_data.loc[rows_indices].reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

# Save results to CSV
output_csv_path = "C:/Users/DFMRendering/Desktop/Oman climate week/Visualization/Maybe_Final/predictions_with_metadata_RF.csv"
results_with_metadata.to_csv(output_csv_path, index=False)

# SHAP analysis
# Use TreeExplainer for optimizing SHAP speed with RandomForest
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

# Check type of shap_values
print(f"Type of shap_values: {type(shap_values)}")  # Added to verify shap_values type

# Save SHAP summary plot without creating a manual figure
shap.summary_plot(shap_values, features, show=False)
shap_summary_path = "C:/Users/DFMRendering/Desktop/Oman climate week/Visualization/Maybe_Final/shap_summary_plot.png"
plt.savefig(shap_summary_path, bbox_inches='tight', dpi=500)
plt.close()  # Use close instead of show to prevent display issues

# Residuals calculation
residuals = np.array(true_values) - np.array(predictions)

# Create hexbin plot for actual vs predicted values with vmax=1000
fig = plt.figure(figsize=(10, 8))  # Removed constrained_layout=True
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.5)

# Main hexbin plot
main_ax = fig.add_subplot(grid[:-1, :-1])
hb = main_ax.hexbin(true_values, predictions, gridsize=20, cmap='Blues', mincnt=8, vmax=1000)  # Changed vmax to 1000
cb = fig.colorbar(hb, ax=main_ax, orientation='vertical')
cb.set_label('Counts', fontsize=12, fontweight='bold')
cb.ax.tick_params(labelsize=11)

# Determine min and max range for both axes to ensure equal scaling
min_val = min(min(true_values), min(predictions))
max_val = max(max(true_values), max(predictions))

# Set equal limits for both axes
main_ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
main_ax.set_xlim([0, max_val])  # Start from 0
main_ax.set_ylim([0, max_val])  # Start from 0
main_ax.tick_params(axis='x', labelsize=11)
main_ax.tick_params(axis='y', labelsize=11)
main_ax.set_xlabel('Actual Values', fontsize=14, fontweight='bold')
main_ax.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')
main_ax.legend(fontsize=12)
main_ax.grid(True, linestyle='--', alpha=0.7)

# Residual histogram on the right
right_ax = fig.add_subplot(grid[:-1, -1])  # Removed 'sharey=main_ax'
right_ax.hist(residuals, bins=30, orientation='horizontal', color='gray', alpha=0.7, edgecolor='black')
right_ax.axhline(0, color='red', linestyle='--')
right_ax.set_title('Residual Histogram', fontsize=14, fontweight='bold')
right_ax.grid(True, linestyle='--', alpha=0.7)
right_ax.tick_params(axis='x', labelsize=11)
right_ax.tick_params(axis='y', labelsize=11)
right_ax.set_ylim([min(residuals), max(residuals)])  # Separate y-axis limit for residuals

# Apply tight_layout to manage spacing
plt.tight_layout()

# Save combined plot
combined_plot_path = "C:/Users/DFMRendering/Desktop/Oman climate week/Visualization/Maybe_Final/combined_hexbin_residual_RF.png"
plt.savefig(combined_plot_path, dpi=500, bbox_inches='tight')
plt.close()  # Use close instead of show to prevent display issues

# Feature Dependence Plots and SHAP vs Target Aggregation
important_features = features.columns[:6]  # Adjust to the top 9 features or modify as needed
num_features = len(important_features)

# Determine number of rows and columns for subplots
cols = 3
rows = math.ceil(num_features / cols)

# Create a GridSpec with an extra row for the colorbar at the bottom
fig = plt.figure(figsize=(20, 5 * rows + 2))  # Increased height for colorbar
gs = gridspec.GridSpec(rows + 1, cols, height_ratios=[1]*rows + [0.05], hspace=0.5, wspace=0.3)

axes = []
all_mean_target = []

for i, feature in enumerate(important_features):
    row = i // cols
    col = i % cols
    ax = fig.add_subplot(gs[row, col])
    
    # Extract SHAP values for the current feature
    shap_feature_values = shap_values[:, features.columns.get_loc(feature)]
    
    # Define bin edges and create bins
    bin_edges = np.linspace(shap_feature_values.min(), shap_feature_values.max(), 4)
    bins = pd.cut(shap_feature_values, bins=bin_edges, include_lowest=True)
    
    # Create a DataFrame for binning
    binned_data = pd.DataFrame({
        'SHAP Value': shap_feature_values,
        'Feature Value': features[feature],
        'Target': target,
        'Bin': bins
    })
    
    # Calculate mean target and mean feature value per bin
    mean_target_per_bin = binned_data.groupby('Bin', observed=False)['Target'].mean()
    mean_feature_per_bin = binned_data.groupby('Bin', observed=False)['Feature Value'].mean()
    
    # Collect all mean target values for normalization
    all_mean_target.extend(mean_target_per_bin.tolist())
    
    # Assign colors based on mean target with normalization
    norm = plt.Normalize(vmin=min(all_mean_target), vmax=25)
    cmap = plt.cm.RdYlBu_r
    colors = cmap(norm(mean_target_per_bin))
    
    # Plot bar chart
    formatted_bin_labels = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in mean_feature_per_bin.index]
    bars = ax.bar(formatted_bin_labels, mean_feature_per_bin, color=colors, edgecolor='black', alpha=0.7)
    
    # Set xtick and ytick label font sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    ax.set_xticks(range(len(mean_feature_per_bin)))
    ax.set_xticklabels(formatted_bin_labels, rotation=0, fontsize=16)
    ax.set_xlabel(f"SHAP Bins", fontsize=16, fontweight='bold')
    ax.set_ylabel(f"Mean {feature}", fontsize=16, fontweight='bold')
    #ax.set_title(f"SHAP Impact on {feature}", fontsize=16, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    axes.append(ax)

# After collecting all mean_target_per_bin, redefine normalization
norm = plt.Normalize(vmin=min(all_mean_target), vmax=25)
cmap = plt.cm.RdYlBu_r

# Re-assign colors with updated normalization
for ax, feature in zip(axes, important_features):
    shap_feature_values = shap_values[:, features.columns.get_loc(feature)]
    bin_edges = np.linspace(shap_feature_values.min(), shap_feature_values.max(), 4)
    bins = pd.cut(shap_feature_values, bins=bin_edges, include_lowest=True)
    binned_data = pd.DataFrame({
        'SHAP Value': shap_feature_values,
        'Feature Value': features[feature],
        'Target': target,
        'Bin': bins
    })
    mean_target_per_bin = binned_data.groupby('Bin', observed=False)['Target'].mean()
    colors = cmap(norm(mean_target_per_bin))
    # Update bar colors
    for bar, color in zip(ax.patches, colors):
        bar.set_facecolor(color)

# Add a single horizontal colorbar at the bottom center
cax = fig.add_subplot(gs[-1, :])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label("Mean Next month Chl-a (mg/m³)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=22)

# Save the matrix plot
matrix_plot_path = "C:/Users/DFMRendering/Desktop/Oman climate week/Visualization/Maybe_Final/matrix_shap_bins_vs_feature_and_target_RF.png"
plt.savefig(matrix_plot_path, dpi=500, bbox_inches='tight')
plt.close()  # Use close instead of show to prevent display issues

# Save analysis to text file with all SHAP and model results
report_path = "C:/Users/DFMRendering/Desktop/Oman climate week/Visualization/Maybe_Final/SHAP_analysis_report_RF.txt"
with open(report_path, "w", encoding="utf-8") as report_file:
    report_file.write("Model Performance Metrics:\n")
    report_file.write(f" - Average MAE: {avg_mae:.2f} ± {std_mae:.2f}\n")
    report_file.write(f" - Average MSE: {avg_mse:.2f} ± {std_mse:.2f}\n")
    report_file.write(f" - Average RMSE: {avg_rmse:.2f} ± {std_rmse:.2f}\n")
    report_file.write(f" - Average R²: {avg_r2:.2f} ± {std_r2:.2f}\n\n")
    
    report_file.write("Feature Importances (Average across folds):\n")
    for index, row in feature_importance_df.iterrows():
        report_file.write(f" - {row['Feature']}: {row['Importance']:.4f}\n")
    report_file.write("\n")
    
    
    report_file.write("Data Details:\n")
    report_file.write(f" - Total Rows Used: {len(modeling_data)}\n")
    report_file.write(f" - Rows Removed Due to Missing Values: {removed_rows}\n\n")
    
    report_file.write("File Outputs:\n")
    report_file.write(f" - CSV File: {output_csv_path}\n")
    report_file.write(f" - SHAP Summary Plot: {shap_summary_path}\n")
    report_file.write(f" - Combined Hexbin and Residual Plot: {combined_plot_path}\n")
    report_file.write(f" - Matrix SHAP Bins Plot: {matrix_plot_path}\n")

print(f"Results saved to {output_csv_path}")
print(f"SHAP summary plot saved to {shap_summary_path}")
print(f"Combined hexbin and residual plot saved to {combined_plot_path}")
print(f"Matrix SHAP bins plot saved to {matrix_plot_path}")
print(f"SHAP analysis report saved to {report_path}")
print("SHAP bins vs feature and target plots saved.")