import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import List, Dict, Any, Tuple
from scipy import stats
from tensorflow_privacy.privacy.analysis import rdp_accountant
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# --------------------------
# PRIVACY ACCOUNTING
# --------------------------

def calculate_epsilon_rdp(config: Dict, data_stats: Dict) -> Tuple[float, float, float]:
    """
    Calculates the final epsilon for the given DP configuration using RDP Accountant.
    
    Returns: epsilon, delta, sigma
    """
    dp_config = config['DIFFERENTIAL_PRIVACY_CONFIG']
    train_config = config['TRAINING_CONFIG']
    
    if not dp_config.get('dp_enabled', False):
        return 0.0, dp_config['dp_delta'], dp_config['dp_sigma'] # Return epsilon=0 if DP is disabled

    # Data stats are now passed from the data processing module
    TOTAL_DATA_POINTS = data_stats['total_train_sequences']
    BATCH_SIZE = train_config['batch_size']
    GLOBAL_ROUNDS = train_config['global_rounds']
    
    # Sampling rate q
    # This assumes that the FL batch size is equivalent to the number of clients participating 
    # per round divided by the total training size, which is complex. 
    # A cleaner assumption (standard in DP-FL papers) is that q = (client_size * batch_size) / (total_data_size) 
    # We use the simpler method from your original code's context, assuming total sequences
    
    # Calculate sampling probability (q)
    # The true complexity of q in FL is (fraction_fit * local_batch_size) / total_data_size.
    # Given the context, we will use the most common DP-SGD definition:
    q = BATCH_SIZE / TOTAL_DATA_POINTS # Assuming a flat DP-SGD update rate across all sequences
    
    # Use parameters from config
    noise_multiplier = dp_config['dp_sigma']
    delta = dp_config['dp_delta']
    steps = GLOBAL_ROUNDS # Each global round is one aggregation step
    
    # Run RDP Accountant
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5, 5., 
               5.5, 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 
               16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 
               26., 27., 28., 29., 30., 32., 34., 36., 38., 40., 
               42., 44., 46., 48., 50., 64, 128, 256, 512, 1024])
               
    rdp = rdp_accountant.compute_rdp(
        q=q,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=orders
    )
    
    epsilon, _, _ = rdp_accountant.get_privacy_spent(orders, rdp, target_delta=delta)
    
    return epsilon, delta, noise_multiplier

# --------------------------
# RESULTS AND PLOTTING
# --------------------------

def print_results_summary(full_results: List[Dict], data_stats: Dict, output_dir: Path):
    """Prints the final summary table and saves it as CSV."""
    df = pd.DataFrame(full_results)
    
    # Calculate Utility Degradation Ratio (RMSE_DP_FL / RMSE_FL)
    fl_rmse_map = df[df['Type'] == 'FL'].set_index('Model')['rmse'].to_dict()
    
    df['Utility Degradation Ratio (RMSE_DP/RMSE_FL)'] = df.apply(
        lambda row: row['rmse'] / fl_rmse_map.get(row['Model'], np.nan) 
        if row['Type'] == 'FL' else np.nan, axis=1
    )
    
    summary_cols = ['Model', 'Type', 'rmse', 'mae', 'Utility Degradation Ratio (RMSE_DP/RMSE_FL)']
    summary_df = df[summary_cols]
    
    logging.info("\n--- FINAL RESULTS SUMMARY ---")
    print(summary_df.to_markdown(index=False, floatfmt=".4f"))
    
    # Save the summary to CSV
    summary_csv_path = output_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Results saved to: {summary_csv_path}")


def plot_metrics_and_ablation(
    full_results: List[Dict], 
    output_dir: Path, 
    epsilon: float, 
    delta: float, 
    sigma: float,
    best_fl_model: Any, 
    X_val: np.ndarray, 
    scaler: Any, 
    all_features: List[str]
):
    """
    Generates all required publication-quality figures:
    1. Multi-Model RMSE Comparison
    2. Utility Degradation Ratio
    3. FL Convergence (RMSE over rounds)
    4. Qualitative Reconstruction Plot (Conv1D-AE)
    5. Residual Distribution Plot (Conv1D-AE)
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(full_results)

    # Filter out Centralized for the ratio calculation
    fl_dp_df = df[df['Type'].isin(['FL', 'DP-FL'])]
    
    # ----------------------------------------------------
    # Plot 1: Multi-Model RMSE Comparison (Bar Chart)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='rmse', hue='Type', data=df)
    plt.title(f'RMSE Comparison Across Architectures and Regimes\n($\epsilon={epsilon:.2f}$, $\delta={delta}$, $\sigma={sigma}$)')
    plt.ylabel('Validation RMSE')
    plt.xlabel('Autoencoder Architecture')
    plt.savefig(output_dir / "multi_model_rmse_comparison.png")
    plt.close()
    logging.info("Figure 1 (RMSE Comparison) saved.")
    
    # ----------------------------------------------------
    # Plot 2: Utility Degradation Ratio (Ablation Plot)
    # ----------------------------------------------------
    ratio_df = fl_dp_df.pivot(index='Model', columns='Type', values='rmse')
    ratio_df['Ratio'] = ratio_df['DP-FL'] / ratio_df['FL']
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=ratio_df.index, y=ratio_df['Ratio'])
    plt.axhline(1.0, color='r', linestyle='--', linewidth=1)
    plt.title(f'Architectural Robustness: RMSE Ratio (DP-FL / FL)')
    plt.ylabel('Utility Degradation Ratio (Closer to 1.0 is better)')
    plt.xlabel('Autoencoder Architecture')
    plt.ylim(0.95, ratio_df['Ratio'].max() * 1.05)
    plt.savefig(output_dir / "performance_ratio_plot.png")
    plt.close()
    logging.info("Figure 2 (Utility Degradation Ratio) saved.")

    # ----------------------------------------------------
    # Plot 3: FL Convergence Plot
    # ----------------------------------------------------
    history_data = []
    for row in df[df['Type'] == 'FL'].itertuples():
        for round_num, rmse in enumerate(row.training_history):
            history_data.append({'Model': row.Model, 'Round': round_num + 1, 'RMSE': rmse})
            
    history_df = pd.DataFrame(history_data)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Round', y='RMSE', hue='Model', data=history_df)
    plt.title('Global Model Convergence Over Federated Rounds')
    plt.xlabel('Communication Round')
    plt.ylabel('Validation RMSE')
    plt.savefig(output_dir / "fl_convergence_plot.png")
    plt.close()
    logging.info("Figure 3 (Convergence Plot) saved.")

    # ----------------------------------------------------
    # Plot 4: Qualitative Reconstruction (Conv1D-AE)
    # ----------------------------------------------------
    # Use the best model (Conv1D-AE) and a single sequence from the validation set
    if best_fl_model is not None:
        
        # Select the first sequence for reconstruction
        sample_seq = X_val[0:1] 
        reconstructed_seq = best_fl_model.predict(sample_seq)[0]
        original_seq = sample_seq[0]
        
        # Inverse transform to plot in original (FHIR) scale
        # We assume the scaler expects a 2D array (samples, features)
        original_features = scaler.inverse_transform(original_seq.reshape(-1, len(all_features)))
        reconstructed_features = scaler.inverse_transform(reconstructed_seq.reshape(-1, len(all_features)))
        
        # Plot only the first 5 continuous features for clarity
        features_to_plot = all_features[:5] 
        
        fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(12, 12), sharex=True)
        
        for i, feature in enumerate(features_to_plot):
            feature_index = all_features.index(feature)
            
            axes[i].plot(original_features[:, feature_index], label='Actual', color='blue', alpha=0.7)
            axes[i].plot(reconstructed_features[:, feature_index], label='Reconstructed', color='red', linestyle='--')
            
            axes[i].set_title(f'Feature: {feature}')
            axes[i].legend(loc='upper right')
        
        fig.suptitle('Qualitative Sequence Reconstruction (Conv1D-AE)', fontsize=16)
        plt.xlabel('Time Step (15-min interval)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(output_dir / "conv1d_sequence_reconstruction.png")
        plt.close()
        logging.info("Figure 4 (Qualitative Reconstruction) saved.")
        
        # ----------------------------------------------------
        # Plot 5: Residual Distribution (Conv1D-AE)
        # ----------------------------------------------------
        # Calculate residuals across the entire validation set
        X_val_pred = best_fl_model.predict(X_val)
        residuals = np.abs(X_val - X_val_pred).flatten()
        
        plt.figure(figsize=(8, 5))
        sns.histplot(residuals, bins=50, kde=True, log_scale=(False, False), color='skyblue')
        plt.title('Distribution of Absolute Reconstruction Residuals (Conv1D-AE)')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency (Log Scale)')
        plt.xlim(0, np.percentile(residuals, 99)) # Clip outliers for better visualization
        plt.savefig(output_dir / "conv1d_residual_distribution.png")
        plt.close()
        logging.info("Figure 5 (Residual Distribution) saved.")

    else:
        logging.warning("Best FL Model (Conv1D-AE) not found. Skipping qualitative plots.")
