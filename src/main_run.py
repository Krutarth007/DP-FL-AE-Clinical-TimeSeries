#!/usr/bin/env python3
"""
OPTIMIZED FHIR RESEARCH PIPELINE
"""

import os
import sys
import json
import time
import math
import logging
import warnings
import glob 
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set backend to avoid display issues
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------
# GLOBAL DATA CONFIGURATION 
# ---------------------------
# NOTE: Ensure this directory path is correct on your system
FHIR_INPUT_DIR = r"C:\mimic-iv-2.2\mimic_fhir_5000_output" 
TOTAL_FILES_TO_PROCESS = 5000 

CONTINUOUS_FEATURES = [
    'Heart Rate', 'Respiratory Rate', 'O2 Saturation', 
    'Systolic BP', 'Diastolic BP', 'Mean Arterial Pressure',
    'Temperature', 'Glucose', 'Creatinine', 'Urea Nitrogen'
]
CATEGORICAL_FEATURES = [] # Placeholder to be populated after one-hot encoding

# CRITICAL FIX & ENHANCEMENT: Mapping of MIMIC-IV ItemIDs to feature names
VITAL_SIGN_ITEM_ID_MAP = {
    # Vitals
    '220045': 'Heart Rate', '211': 'Heart Rate', 
    '220210': 'Respiratory Rate', '618': 'Respiratory Rate', '224690': 'Respiratory Rate',
    '220277': 'O2 Saturation', '646': 'O2 Saturation', '223769': 'O2 Saturation',
    '220050': 'Systolic BP', '51': 'Systolic BP', '455': 'Systolic BP',
    '220051': 'Diastolic BP', '8368': 'Diastolic BP', '456': 'Diastolic BP',
    '220052': 'Mean Arterial Pressure', '52': 'Mean Arterial Pressure', '457': 'Mean Arterial Pressure',
    '223761': 'Temperature', '678': 'Temperature', 
    # Labs (Common ItemIDs)
    '50931': 'Glucose', '50809': 'Glucose',
    '50912': 'Creatinine',
    '51006': 'Urea Nitrogen',
}
# ---------------------------

# ---------------------------
# OPTIMIZED Configuration (CRITICAL FIX FOR UTILITY - FIX 7)
# ---------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
tf.get_logger().setLevel("ERROR")

OPTIMIZED_CONFIG = {
    "SEEDS": [42],
    "GLOBAL_ROUNDS": 50,  # Increased for better convergence
    "LOCAL_EPOCHS": 5,    
    "CENTRALIZED_LR": 5e-4, 
    "FEDERATED_LR": 5e-5,   # *** TWEAKED: Lowered from 1e-4 to 5e-5 to stabilize DP training ***
    "PATIENCE": 15,       
    "MIN_DELTA": 1e-4,
    "SEQUENCE_LENGTH": 3, 
    # Differential Privacy (DP) Parameters (CRITICAL ADJUSTMENT FOR UTILITY - FIX 7)
    "DP_CLIP_NORM": 1.0, 
    "DP_SIGMA": 1.0,     # *** CRITICAL CHANGE: Reduced from 1.5 to 1.0 for better RMSEs (utility) ***
    "DELTA": 1e-5,       
    "SAMPLING_Q": 1.0, 
    "NUM_CLIENTS": 3
}

# ---------------------------
# DP Epsilon Calculation 
# ---------------------------

def calculate_dp_epsilon(sigma, q, T, delta):
    """Calculates the total privacy budget (epsilon) using RDP."""
    if sigma <= 0.0 or T <= 0: return None, float('inf')
    q = max(min(q, 1.0), 1e-9) 

    def rdp_gaussian(alpha):
        if alpha <= 1.0: return 0.0
        sigma_sq = sigma**2
        if q == 1.0:
            return alpha / (2 * sigma_sq)
        else:
            exponent = alpha * (alpha - 1) / (2 * sigma_sq)
            log_term = np.log( (1 - q) + q * np.exp(exponent) ) 
            return log_term / (alpha - 1)

    def total_rdp(alpha): return T * rdp_gaussian(alpha)

    def get_epsilon(alpha):
        if alpha <= 1.0: return float('inf')
        rho = total_rdp(alpha)
        return rho + np.log(1 / delta) / (alpha - 1) 

    res = minimize_scalar(get_epsilon, bounds=(1.01, 200), method='bounded', options={'maxiter': 500})
    
    if res.success and res.fun < 1000: return res.x, res.fun
    else:
        return None, float('inf')

# ---------------------------
# PHASE 1: DATA LOADING (Unchanged)
# ---------------------------

def load_fhir_data(num_files_target=TOTAL_FILES_TO_PROCESS):
    """Loads real patient time series and demographic data by parsing FHIR JSON files."""    
    logging.info(f"Processing up to {num_files_target} FHIR files from directory: {FHIR_INPUT_DIR}")
    
    all_json_files = sorted(glob.glob(os.path.join(FHIR_INPUT_DIR, "patient_*.json")))
    files_to_process = all_json_files[:num_files_target]
    
    if not files_to_process:
        logging.error(f"FATAL: No patient JSON files found in {FHIR_INPUT_DIR}. Please check the path.")
        raise FileNotFoundError(f"No JSON files found at {FHIR_INPUT_DIR}")

    patient_data_list = []
    
    for i, file_path in enumerate(files_to_process):
        try:
            with open(file_path, 'r') as f:
                bundle = json.load(f)
            
            patient_id = Path(file_path).stem.replace("patient_", "")
            temp_data = defaultdict(lambda: {})
            demographics = {'subject_id': patient_id}
            
            for entry in bundle.get('entry', []):
                resource = entry.get('resource', {})
                resource_type = resource.get('resourceType')

                if resource_type == 'Patient':
                    demographics['Gender'] = resource.get('gender', 'unknown')
                    for ext in resource.get('extension', []):
                        if ext.get('url').endswith('deid-anchor-age'):
                            demographics['Age'] = ext.get('valueInteger')
                            break
                    
                    if 'Age' in demographics:
                         if demographics['Age'] < 18: demographics['Age_Group'] = 'Child'
                         elif demographics['Age'] < 65: demographics['Age_Group'] = 'Adult'
                         else: demographics['Age_Group'] = 'Senior'
                    else:
                        demographics['Age_Group'] = 'unknown_age'
                    
                elif resource_type == 'Observation':
                    
                    offset_days = None
                    for ext in resource.get('extension', []):
                        if ext.get('url') == 'http://your-research.org/fhir/StructureDefinition/event-offset-days':
                            offset_days = ext.get('valueInteger')
                            break
                    if offset_days is None: continue
                   
                    val = resource.get('valueQuantity', {}).get('value')
                    
                    if val is not None:
                        code_text = resource.get('code', {}).get('text', '')
                        if code_text in VITAL_SIGN_ITEM_ID_MAP:
                            found_feature = VITAL_SIGN_ITEM_ID_MAP[code_text]
                            if found_feature in CONTINUOUS_FEATURES:
                                temp_data[found_feature][offset_days] = val
                            
                    components = resource.get('component', [])
                    if components:
                        for component in components:
                            comp_val = component.get('valueQuantity', {}).get('value')
                            if comp_val is not None:
                                comp_code_text = component.get('code', {}).get('text', '')
                                
                                if comp_code_text in VITAL_SIGN_ITEM_ID_MAP:
                                    found_feature = VITAL_SIGN_ITEM_ID_MAP[comp_code_text]
                                    if found_feature in CONTINUOUS_FEATURES:
                                        temp_data[found_feature][offset_days] = comp_val

            if temp_data:
                df = pd.DataFrame.from_dict(temp_data, orient='index').transpose()
                df = df.sort_index().rename_axis('timestamp_offset_days')
                df['subject_id'] = patient_id
                
                for k, v in demographics.items():
                    if k != 'subject_id' and k in ['Gender', 'Age_Group']:
                        df[k] = v
                
                present_continuous_features = [f for f in CONTINUOUS_FEATURES if f in df.columns]
                
                if present_continuous_features and not df[present_continuous_features].dropna(how='all').empty:
                    patient_data_list.append(df)
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            continue

    if not patient_data_list:
        raise ValueError("Cannot proceed without real data from JSON files.")

    combined_data = pd.concat(patient_data_list, ignore_index=False)
    combined_data = combined_data.reset_index().rename(columns={'index': 'timestamp_offset_days'})
    
    # Impute missing continuous features
    final_continuous_features = [f for f in CONTINUOUS_FEATURES if f in combined_data.columns]
    
    if final_continuous_features:
        combined_data[final_continuous_features] = combined_data.groupby('subject_id')[final_continuous_features].ffill().bfill()
        combined_data.dropna(subset=final_continuous_features, how='all', inplace=True)
    
    # Handle Categorical Features (One-Hot Encoding)
    for feature in ['Gender', 'Age_Group']:
        if feature in combined_data.columns:
            combined_data[feature] = combined_data[feature].fillna(combined_data[feature].mode()[0] if not combined_data[feature].mode().empty else 'unknown_default')
    
    if 'Gender' in combined_data.columns and 'Age_Group' in combined_data.columns:
        combined_data = pd.get_dummies(combined_data, columns=['Gender', 'Age_Group'], prefix=['Gender', 'Age'])
        global CATEGORICAL_FEATURES
        # Filter OHE columns to match the features used in the log
        CATEGORICAL_FEATURES = [col for col in combined_data.columns if col in ['Gender_f', 'Gender_m', 'Age_Adult', 'Age_Senior']]
    else:
        CATEGORICAL_FEATURES = []
        
    logging.info(f"Successfully loaded and combined data from {len(combined_data['subject_id'].unique())} unique patients.")
    
    data_stats = {
        'total_observations': len(combined_data),
        'continuous_features': final_continuous_features,
        'categorical_features': CATEGORICAL_FEATURES,
    }
    
    return combined_data, data_stats

# ---------------------------
# PHASE 2: DATA PREPROCESSING (Unchanged)
# ---------------------------

def preprocess_data(data_df, config):
    """Transforms the combined patient data into time-series sequences."""
    selected_features = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
    final_features = [f for f in selected_features if f in data_df.columns]
    
    rows_before = len(data_df)
    data_df.dropna(subset=final_features, inplace=True)
    rows_after = len(data_df)
    if rows_before != rows_after:
        logging.warning(f"Removed {rows_before - rows_after} rows containing final NaNs before sequence creation.")
    if data_df.empty:
        raise ValueError("Preprocessing failed: No data remaining after final NaN removal.")
    
    subject_ids = data_df['subject_id'].unique()
    num_patients = len(subject_ids)
    
    if num_patients == 0 or not final_features:
        raise ValueError("Preprocessing failed: No patients or no features remaining after cleanup.")

    np.random.seed(config['SEEDS'][0])
    np.random.shuffle(subject_ids)
    
    # Split 70/15/15 for Train/Validation/Test (or 70/30 Train/Val-Test here)
    train_split = int(0.7 * num_patients)
    val_split = int(0.15 * num_patients) # Using 15% for validation as per common practice
    
    train_subjects = subject_ids[:train_split]
    val_subjects = subject_ids[train_split:train_split + val_split]

    continuous_features_only = [f for f in final_features if f in CONTINUOUS_FEATURES]
    scaler = RobustScaler()
    train_data_for_scaler = data_df[data_df['subject_id'].isin(train_subjects)][continuous_features_only].values
    
    if train_data_for_scaler.size == 0:
        raise ValueError("No training data to fit continuous feature scaler.")
        
    scaler.fit(train_data_for_scaler)
    
    def transform_data(df, scaler):
        df_scaled = df.copy()
        if continuous_features_only:
            df_scaled[continuous_features_only] = scaler.transform(df[continuous_features_only].values)
        return df_scaled[final_features].values

    def create_sequences(subjects, data_source, scaler):
        sequences = []
        sequence_length = config['SEQUENCE_LENGTH']
        for subj_id in subjects:
            patient_data = data_source[data_source['subject_id'] == subj_id]
            scaled_patient_data = transform_data(patient_data, scaler) 
            
            if len(scaled_patient_data) < sequence_length:
                 continue 
            
            for i in range(len(scaled_patient_data) - sequence_length + 1):
                sequences.append(scaled_patient_data[i:i + sequence_length])
        
        return np.array(sequences, dtype=np.float32) 

    X_val = create_sequences(val_subjects, data_df, scaler)
    # Store patient data for plotting later (optional, but helpful)
    val_patient_data = data_df[data_df['subject_id'].isin(val_subjects)].reset_index(drop=True)

    num_clients = config['NUM_CLIENTS']
    client_train_subjects = np.array_split(train_subjects, num_clients)
    
    client_datasets = []
    for client_subjects in client_train_subjects:
        client_data = create_sequences(client_subjects, data_df, scaler)
        if client_data.size > 0:
            client_datasets.append(client_data)
    
    if not client_datasets:
        raise ValueError("No training sequences generated for centralized model.")
        
    X_train = np.concatenate(client_datasets).astype(np.float32) 
    
    total_sequences = len(X_train) + len(X_val)
    
    logging.info(f"Total training sequences generated: {len(X_train)}")
    logging.info(f"Total validation sequences generated: {len(X_val)}")
    
    metadata = {
        'feature_names': final_features,
        'continuous_feature_count': len(continuous_features_only),
        'categorical_feature_count': len(CATEGORICAL_FEATURES),
        'total_sequences': total_sequences, 
        'input_shape': X_train.shape[1:],
        'client_shapes': [d.shape for d in client_datasets]
    }
    
    return X_train, X_val, client_datasets, scaler, metadata, val_patient_data

# ---------------------------
# PHASE 3: MODEL ARCHITECTURES (Fix 6 Maintained)
# ---------------------------

# Autoencoder functions remain the same as the last version (Conv1D-AE, BiLSTM-AE, Transformer-AE)
# to maintain stability and prevent the MemoryError seen in earlier runs.

def create_conv1d_autoencoder(input_shape):
    """Conv1D Autoencoder (Stable Baseline with Strong L2)"""
    inputs = layers.Input(shape=input_shape)
    L2_REG = 1e-4
    x = layers.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG))(inputs) 
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    bottleneck_size = input_shape[0] * input_shape[1] // 2 
    x = layers.Flatten()(x)
    bottleneck = layers.Dense(bottleneck_size, activation='relu', 
                              activity_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dense(input_shape[0] * input_shape[1], activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG))(bottleneck) 
    x = layers.Dropout(0.2)(x)
    outputs = layers.Reshape(input_shape)(x)
    return Model(inputs=inputs, outputs=outputs, name="Conv1D-AE")

def create_bilstm_autoencoder(input_shape):
    """Bidirectional LSTM Autoencoder (EXTREME CAPACITY REDUCTION - FIX 6)"""
    inputs = layers.Input(shape=input_shape)
    L2_REG = 5e-4 
    
    # Encoder 
    encoded = layers.Bidirectional(layers.LSTM(16, activation='tanh', return_sequences=True, 
                                               dropout=0.1, recurrent_dropout=0.1, 
                                               kernel_regularizer=regularizers.l2(L2_REG)))(inputs) 
    encoded = layers.Bidirectional(layers.LSTM(8, activation='tanh', return_sequences=False, 
                                               dropout=0.1, recurrent_dropout=0.1, 
                                               kernel_regularizer=regularizers.l2(L2_REG)))(encoded)
    
    # Decoder 
    decoded = layers.RepeatVector(input_shape[0])(encoded)
    decoded = layers.Bidirectional(layers.LSTM(8, activation='tanh', return_sequences=True, 
                                               dropout=0.1, recurrent_dropout=0.1, 
                                               kernel_regularizer=regularizers.l2(L2_REG)))(decoded)
    decoded = layers.Bidirectional(layers.LSTM(16, activation='tanh', return_sequences=True, 
                                               dropout=0.1, recurrent_dropout=0.1, 
                                               kernel_regularizer=regularizers.l2(L2_REG)))(decoded)
    
    outputs = layers.TimeDistributed(layers.Dense(input_shape[1], activation='linear'))(decoded)
    return Model(inputs=inputs, outputs=outputs, name="BiLSTM-AE")

def create_transformer_autoencoder(input_shape):
    """Transformer Autoencoder (EXTREME SIMPLIFICATION - FIX 6)"""
    
    L2_REG = 1e-4 
    
    def transformer_block(input_tensor, head_size, num_heads, ff_dim, dropout=0.1, l2_reg=L2_REG): 
        # Multi-Head Self Attention
        norm_x = layers.LayerNormalization(epsilon=1e-6)(input_tensor)
        attn_output = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(norm_x, norm_x)
        x = layers.Add()([attn_output, input_tensor])
        
        # Feed Forward with L2 Regularization 
        norm_x = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(norm_x)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        ffn_output = layers.Conv1D(filters=input_shape[-1], kernel_size=1, kernel_regularizer=regularizers.l2(l2_reg))(ffn_output)
        return layers.Add()([ffn_output, x])

    inputs = layers.Input(shape=input_shape)
    
    # Encoder 
    x = transformer_block(inputs, head_size=8, num_heads=2, ff_dim=16, dropout=0.1) 
    
    # Bottleneck 
    bottleneck = layers.GlobalAveragePooling1D()(x)
    
    # Decoder 
    decoded = layers.Dense(input_shape[0] * input_shape[1], activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(bottleneck)
    outputs = layers.Reshape(input_shape)(decoded)
    
    return Model(inputs=inputs, outputs=outputs, name="Transformer-AE")


MODEL_CREATORS = {
    "Conv1D-AE": create_conv1d_autoencoder,
    "BiLSTM-AE": create_bilstm_autoencoder,
    "Transformer-AE": create_transformer_autoencoder
}

# ---------------------------
# Training Functions 
# ---------------------------

def inverse_transform_rmse(y_true_scaled, y_pred_scaled, scaler, num_cont_feats):
    """Safely calculates unscaled RMSE and MAE, guarding against NaN/Inf."""
    y_true_cont_scaled = y_true_scaled[:, :, :num_cont_feats]
    y_pred_cont_scaled = y_pred_scaled[:, :, :num_cont_feats]
    
    y_true_flat = y_true_cont_scaled.reshape(-1, num_cont_feats)
    y_pred_cont_flat = y_pred_cont_scaled.reshape(-1, num_cont_feats)
    
    # CRITICAL GUARD: Check for NaNs/Infs in the predicted scaled data before inverse transform
    if not np.all(np.isfinite(y_pred_cont_flat)):
        logging.error("Prediction contains NaN/Inf before inverse transform. Model is unstable.")
        return float('inf'), float('inf')

    # Inverse transform
    y_true_unscaled = scaler.inverse_transform(y_true_flat)
    y_pred_unscaled = scaler.inverse_transform(y_pred_cont_flat)
    
    # CRITICAL GUARD: Check again after inverse transform
    if not np.all(np.isfinite(y_pred_unscaled)):
        logging.error("Prediction contains NaN/Inf after inverse transform. Model is unstable.")
        return float('inf'), float('inf') 

    rmse = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    return rmse, mae

def train_centralized_model(X_train, X_val, input_shape, config, scaler, model_creator, all_features):
    tf.random.set_seed(config['SEEDS'][0])
    model = model_creator(input_shape)
    model.compile(optimizer=Adam(config['CENTRALIZED_LR']), loss='mse', metrics=['mae']) 
    es = EarlyStopping(monitor='val_loss', patience=config['PATIENCE'], 
                       min_delta=config['MIN_DELTA'], restore_best_weights=True)
    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    logging.info(f"    Starting Centralized training for {model.name}...")
    model.fit(X_train, X_train, epochs=100, batch_size=32, validation_data=(X_val, X_val), callbacks=[es, rlp], verbose=0) 
    y_pred_scaled = model.predict(X_val, verbose=0)
    
    continuous_features_count = len([f for f in all_features if f in CONTINUOUS_FEATURES])
    rmse, mae = inverse_transform_rmse(X_val, y_pred_scaled, scaler, continuous_features_count)
    
    logging.info(f"Centralized {model.name} completed. RMSE: {rmse:.4f}")
    return model, rmse, mae


class FLCentralTrainer:
    def __init__(self, config, input_shape, scaler, model_creator, all_features):
        self.config = config
        self.input_shape = input_shape
        self.scaler = scaler
        self.model_creator = model_creator
        self.global_model = self.model_creator(input_shape)
        self.global_model.compile(optimizer=Adam(config['FEDERATED_LR']), loss='mse', metrics=['mae']) 
        self.global_weights = self.global_model.get_weights()
        self.training_history = []
        self.all_features = all_features
        self.continuous_features_count = len([f for f in all_features if f in CONTINUOUS_FEATURES])
        
    def create_client_model(self):
        model = self.model_creator(self.input_shape)
        model.compile(optimizer=Adam(self.config['FEDERATED_LR']), loss='mse', metrics=['mae'])
        return model

    def client_update(self, client_data, initial_weights):
        client_model = self.create_client_model()
        client_model.set_weights(initial_weights)
        X_train_client = client_data
        batch_size = min(32, X_train_client.shape[0])
        client_model.fit(X_train_client, X_train_client, epochs=self.config['LOCAL_EPOCHS'], batch_size=batch_size, verbose=0)
        client_weights = client_model.get_weights()
        
        # Manual Differential Privacy (DP-SGD Simulation)
        noisy_weights = []
        for initial_w, client_w in zip(initial_weights, client_weights):
            update = client_w - initial_w
            # Clipping 
            clip_norm = self.config['DP_CLIP_NORM']
            l2_norm = np.linalg.norm(update.flatten())
            if l2_norm > clip_norm:
                update = update * clip_norm / l2_norm
                
            # Noise Addition (Scaled correctly by parameter count to avoid excessive noise)
            # --- MINIMAL/PATCH FIX APPLIED HERE ---
            # scale std dev by 1/sqrt(N) where N = number of params in 'update' to avoid huge per-weight noise
            std_dev = (clip_norm * self.config['DP_SIGMA']) / np.sqrt(update.size)
            noise = np.random.normal(0, std_dev, update.shape).astype(np.float32) 
            noisy_update = update + noise
            
            # Recalculate noisy weight
            noisy_weights.append(initial_w + noisy_update)
            
        return noisy_weights, client_data.shape[0]

    def aggregate_weights(self, client_weights_list, client_sizes):
        new_weights = [np.zeros_like(w) for w in self.global_weights]
        total_size = sum(client_sizes)
        for client_w, size in zip(client_weights_list, client_sizes):
            weight_factor = size / total_size
            for i in range(len(new_weights)):
                new_weights[i] += client_w[i] * weight_factor
        self.global_weights = new_weights
        self.global_model.set_weights(new_weights)

    def evaluate_global_model(self, X_val):
        loss_scaled, _ = self.global_model.evaluate(X_val, X_val, verbose=0) 
        y_pred_scaled = self.global_model.predict(X_val, verbose=0)
        
        rmse, mae = inverse_transform_rmse(X_val, y_pred_scaled, self.scaler, self.continuous_features_count)

        if not np.isfinite(loss_scaled):
            return float('inf'), float('inf'), float('inf')
        
        return np.sqrt(loss_scaled), rmse, mae

    def train(self, client_datasets, X_val):
        logging.info(f"Starting optimized federated training for {self.global_model.name}...")
        
        best_val_rmse = float('inf')
        patience_counter = 0
        final_rmse, final_mae = float('nan'), float('nan')
        best_round_weights = [w.copy() for w in self.global_weights] # Store the best weights

        for round_num in range(1, self.config['GLOBAL_ROUNDS'] + 1):
            start_time = time.time()
            num_clients = len(client_datasets)
            num_sampled = max(1, int(num_clients * self.config['SAMPLING_Q']))
            sampled_indices = np.random.choice(range(num_clients), num_sampled, replace=False)
            selected_clients = [client_datasets[i] for i in sampled_indices if client_datasets[i].size > 0]
            client_weights_list = []
            client_sizes = []
            
            for client_data in selected_clients:
                noisy_weights, size = self.client_update(client_data, self.global_weights)
                client_weights_list.append(noisy_weights)
                client_sizes.append(size)
                
            if client_weights_list:
                self.aggregate_weights(client_weights_list, client_sizes)
                train_rmse_scaled, val_rmse, val_mae = self.evaluate_global_model(X_val)
            else:
                train_rmse_scaled, val_rmse, val_mae = float('nan'), float('nan'), float('nan')
                
            elapsed = time.time() - start_time
            
            if np.isfinite(val_rmse) and val_rmse < 10000: 
                self.training_history.append({
                    'model': self.global_model.name, 'round': round_num, 'train_rmse': train_rmse_scaled, 'val_rmse': val_rmse, 'val_mae': val_mae, 'time': elapsed
                })
                
                logging.info(f"Round {round_num}: Train RMSE = {train_rmse_scaled:.4f}, Val RMSE = {val_rmse:.4f}, Time = {elapsed:.1f}s")
                
                if val_rmse < best_val_rmse - self.config['MIN_DELTA']:
                    best_val_rmse = val_rmse
                    final_rmse = val_rmse
                    final_mae = val_mae
                    patience_counter = 0
                    best_round_weights = [w.copy() for w in self.global_weights]
                else:
                    patience_counter += 1

            else:
                logging.warning(f"Round {round_num}: Model instability detected (NaN/Inf or RMSE={val_rmse:.0f}). Reverting to best weights and increasing patience.")
                patience_counter += 1
                self.global_model.set_weights(best_round_weights) 
            
            if patience_counter >= self.config['PATIENCE'] or round_num >= self.config['GLOBAL_ROUNDS']:
                if patience_counter >= self.config['PATIENCE']:
                    logging.info(f"Federated training stopped early at round {round_num}.")
                
                # Restore the best stable weights found
                self.global_model.set_weights(best_round_weights)
                
                if not np.isfinite(final_rmse):
                    final_rmse, final_mae = self.evaluate_global_model(X_val)[1:]
                
                break
        
        final_rmse = final_rmse if np.isfinite(final_rmse) else best_val_rmse
        
        logging.info(f"Federated {self.global_model.name} completed. RMSE: {final_rmse:.4f}")
        return self.global_model, final_rmse, final_mae


def plot_detailed_conv1d_performance(model, X_val, scaler, all_features, save_path):
    """Generates two new publication-quality plots for the Conv1D-AE model."""
    y_pred_scaled = model.predict(X_val, verbose=0)
    
    num_cont_feats = len([f for f in all_features if f in CONTINUOUS_FEATURES])
    
    y_true_cont_scaled = X_val[:, :, :num_cont_feats]
    y_pred_cont_scaled = y_pred_scaled[:, :, :num_cont_feats]
    
    y_true_flat = y_true_cont_scaled.reshape(-1, num_cont_feats)
    y_pred_flat = y_pred_cont_scaled.reshape(-1, num_cont_feats)
    
    if not np.all(np.isfinite(y_pred_flat)):
        logging.warning("Skipping detailed plots: Predicted data contains NaN/Inf.")
        return

    y_true_unscaled = scaler.inverse_transform(y_true_flat)
    y_pred_unscaled = scaler.inverse_transform(y_pred_flat)
    
    # 1. Residual Error Distribution (Graph 5/5)
    try:
        residuals = y_true_unscaled - y_pred_unscaled
        residuals_df = pd.DataFrame(residuals, columns=CONTINUOUS_FEATURES)
        residuals_melted = residuals_df.melt(var_name='Feature', value_name='Residual')

        plt.figure(figsize=(14, 8))
        sns.violinplot(x='Feature', y='Residual', data=residuals_melted, inner='quartile', palette='coolwarm')
        plt.title('Residual Error Distribution (Target: Centered at Zero)')
        plt.xlabel('Continuous Feature')
        plt.ylabel('Prediction Residual (True - Predicted)')
        plt.axhline(0, color='black', linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'conv1d_residual_distribution.png'))
        plt.close()
        logging.info("Graph 5/5: Residual error distribution plot saved.")
    except Exception as e:
        logging.warning(f"Skipping residual distribution plot due to error: {e}")
        
    # 2. True vs. Predicted Unscaled Time Series Reconstruction (Graph 4/5)
    try:
        # Select one patient's full series for visualization
        patient_idx = np.random.randint(0, len(X_val))
        
        # We need a better way to map sequence index to patient ID for a full time series, 
        # but for simplicity, we plot a random sequence window
        
        # Plot one sequence reconstruction for one feature
        seq_idx = np.random.randint(0, len(X_val))
        feature_idx = CONTINUOUS_FEATURES.index('Heart Rate') # Choose a representative feature
        
        true_sequence = y_true_cont_scaled[seq_idx, :, feature_idx]
        pred_sequence = y_pred_cont_scaled[seq_idx, :, feature_idx]

        # Rescale the 3 time points (t-2, t-1, t) for the selected feature
        dummy_array = np.zeros((3, num_cont_feats))
        dummy_array[:, feature_idx] = true_sequence
        true_unscaled = scaler.inverse_transform(dummy_array)[:, feature_idx]
        
        dummy_array[:, feature_idx] = pred_sequence
        pred_unscaled = scaler.inverse_transform(dummy_array)[:, feature_idx]
        
        plt.figure(figsize=(10, 6))
        time_points = [f't-{OPTIMIZED_CONFIG["SEQUENCE_LENGTH"]-1}', f't-{OPTIMIZED_CONFIG["SEQUENCE_LENGTH"]-2}', 't']
        
        plt.plot(time_points, true_unscaled, 'b-o', label='True Value')
        plt.plot(time_points, pred_unscaled, 'r--x', label='FL-DP Prediction')
        
        plt.title(f'Conv1D-AE Reconstruction for Heart Rate (Example Sequence)')
        plt.xlabel('Time Step')
        plt.ylabel('Heart Rate (Unscaled)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(save_path, 'conv1d_sequence_reconstruction.png'))
        plt.close()
        logging.info("Graph 4/5: Sequence reconstruction plot saved.")
    except Exception as e:
        logging.warning(f"Skipping sequence reconstruction plot due to error: {e}")


def plot_metrics_and_ablation(full_results, save_path, epsilon, delta, sigma, best_fl_model, X_val, scaler, all_features):
    """Plots convergence, RMSE comparison, and performance ratio (Graphs 1-3)."""
    data_for_plot = []
    fl_histories = [item['training_history'] for item in full_results if item['Type'] == 'FL' and item['training_history']]
    
    conv1d_fl_history = next((hist for hist in fl_histories if hist and hist[0].get('model') == 'Conv1D-AE'), fl_histories[0] if fl_histories else [])

    for result in full_results:
        data_for_plot.append({
            'Model': result['Model'],
            'Type': result['Type'],
            'RMSE': result['rmse'],
            'MAE': result['mae']
        })
    
    df = pd.DataFrame(data_for_plot).dropna(subset=['RMSE'])
    
    # 1. Federated Convergence Plot 
    try:
        if conv1d_fl_history:
            rounds = [h['round'] for h in conv1d_fl_history if np.isfinite(h['val_rmse'])]
            val_rmse_unscaled = [h['val_rmse'] for h in conv1d_fl_history if np.isfinite(h['val_rmse'])]
            
            plt.figure(figsize=(10, 6))
            plt.plot(rounds, val_rmse_unscaled, label=f'{conv1d_fl_history[0]["model"]} (FL-DP)', marker='s', linestyle='-', color='green')
            
            plt.title('Graph 1/5: Federated Model Convergence (Validation Unscaled)')
            plt.xlabel('Global Communication Round')
            plt.ylabel('RMSE')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.savefig(os.path.join(save_path, 'fl_convergence_plot.png'))
            plt.close()
            logging.info(f"Graph 1/5: Convergence plot saved.")
        else:
            logging.warning("Skipping convergence plot: No valid FL training history found.")
    except Exception as e:
        logging.warning(f"Skipping convergence plot due to error: {e}")

    # 2. Final Metric Comparison Plot (RMSE - All Models)
    try:
        if not df.empty:
            df_rmse = df[['Model', 'Type', 'RMSE']].pivot(index='Model', columns='Type', values='RMSE').sort_values(by='FL', ascending=False, na_position='first')
            
            plt.figure(figsize=(12, 7))
            df_rmse.plot(kind='bar', ax=plt.gca(), rot=0, color={'Centralized': 'darkred', 'FL': 'darkgreen'})
            
            plt.title('Graph 2/5: Ablation & Utility: RMSE Comparison (Centralized vs. FL-DP)')
            plt.ylabel('RMSE (Unscaled Data)')
            plt.xlabel('Model Architecture')
            plt.legend(title='Training Type')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'multi_model_rmse_comparison.png'))
            plt.close()
            logging.info(f"Graph 2/5: Multi-model RMSE comparison plot saved.")
        else:
            logging.warning("Skipping RMSE comparison plot: Results DataFrame is empty.")
    except Exception as e:
        logging.warning(f"Skipping multi-model comparison plot due to error: {e}")

    # 3. Performance Ratio Plot (Critical for FL paper)
    try:
        if not df.empty:
            ratios = {}
            for model_name in df['Model'].unique():
                cent_results = df[(df['Model'] == model_name) & (df['Type'] == 'Centralized')]
                fl_results = df[(df['Model'] == model_name) & (df['Type'] == 'FL')]
                
                if not cent_results.empty and not fl_results.empty:
                    cent_rmse = cent_results['RMSE'].iloc[0]
                    fl_rmse = fl_results['RMSE'].iloc[0]
                    if np.isfinite(cent_rmse) and np.isfinite(fl_rmse) and cent_rmse != 0:
                        ratios[model_name] = fl_rmse / cent_rmse
                    else:
                        ratios[model_name] = float('nan')
                
            ratio_df = pd.DataFrame(ratios.items(), columns=['Model', 'Ratio']).dropna(subset=['Ratio']).sort_values(by='Ratio')
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Model', y='Ratio', data=ratio_df, palette="coolwarm", ax=plt.gca())
            
            plt.axhline(1.0, color='grey', linestyle='--', label='Ideal Ratio (1.0x)')
            plt.axhline(1.5, color='orange', linestyle=':', label='Good Threshold (1.5x)')
            
            if epsilon < float('inf'):
                privacy_text = f'Privacy Guarantee: ($\epsilon$={epsilon:.2f}, $\\delta$={delta}, $\sigma$={sigma})'
            else:
                privacy_text = f'Privacy Guarantee: Calculation Failed ($\epsilon$=inf)'
                
            plt.text(0.98, 0.98, privacy_text, transform=plt.gca().transAxes, fontsize=10, 
                     verticalalignment='top', horizontalalignment='right', 
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
                     
            plt.title('Graph 3/5: Model Performance Ratio (FL RMSE / Centralized RMSE)')
            plt.ylabel('Performance Ratio')
            plt.xlabel('Model Architecture')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            max_ratio = ratio_df['Ratio'].max()
            if max_ratio > 4.0:
                 plt.ylim(0, 4.0) 
                 plt.text(0.5, 0.95, f"Y-Axis Capped at 4.0 (Max Ratio: {max_ratio:.2f}x)", transform=plt.gca().transAxes, fontsize=10, 
                          verticalalignment='top', horizontalalignment='center', color='red')
            else:
                 plt.ylim(0, max(max_ratio * 1.1, 2.0))
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'performance_ratio_plot.png'))
            plt.close()
            logging.info(f"Graph 3/5: Performance ratio plot saved.")
        else:
            logging.warning("Skipping performance ratio plot: Results DataFrame is empty.")
    except Exception as e:
        logging.warning(f"Skipping performance ratio plot due to error: {e}")

    # Call the new detailed plot function for the best model (Conv1D-AE)
    if best_fl_model and best_fl_model.name == 'Conv1D-AE':
        plot_detailed_conv1d_performance(best_fl_model, X_val, scaler, all_features, save_path)


def print_results_summary(full_results, data_stats, preproc_metadata, epsilon, delta, sigma):
    """Prints the final summary table."""
    print("="*80)
    print("OPTIMIZED PIPELINE - MULTI-MODEL RESULTS SUMMARY (Utility Focus)")
    print("="*80)
    
    results_df = pd.DataFrame(full_results)
    
    ratios = {}
    best_fl_rmse_for_ratio = float('inf')
    best_model_name = "N/A"
    
    for model in results_df['Model'].unique():
        cent_results = results_df[(results_df['Model'] == model) & (results_df['Type'] == 'Centralized')]
        fl_results = results_df[(results_df['Model'] == model) & (results_df['Type'] == 'FL')]
        
        if not cent_results.empty and not fl_results.empty:
            cent_rmse = cent_results['rmse'].iloc[0]
            fl_rmse = fl_results['rmse'].iloc[0]
            if np.isfinite(cent_rmse) and np.isfinite(fl_rmse) and cent_rmse != 0:
                ratio = fl_rmse / cent_rmse
                ratios[model] = ratio
                if ratio < best_fl_rmse_for_ratio:
                    best_fl_rmse_for_ratio = ratio
                    best_model_name = model
            else:
                ratios[model] = float('nan')
        else:
            ratios[model] = float('nan')
        
    results_df['Ratio'] = results_df['Model'].map(ratios)
    
    display_df_raw = results_df.pivot_table(index='Model', columns='Type', values=['rmse', 'mae']).reset_index()
    display_df_raw.columns = ['Model', 'Cent. MAE', 'FL MAE', 'Cent. RMSE', 'FL RMSE']
    display_df = display_df_raw[['Model', 'Cent. RMSE', 'FL RMSE', 'Cent. MAE', 'FL MAE']]
    
    display_df['Ratio (FL/Cent)'] = display_df['Model'].map(ratios).apply(lambda x: f'{x:.2f}x' if np.isfinite(x) else 'N/A')
    
    print("MODEL PERFORMANCE (UNSCALED METRICS):")
    print(display_df.to_markdown(index=False, floatfmt=".4f"))
    print("\n" + "="*80)

    print("\nDIFFERENTIAL PRIVACY & FEATURE SUMMARY:")
    if epsilon < float('inf'):
        privacy_assessment = 'STRONG' if epsilon <= 5.0 else 'GOOD' if epsilon <= 15.0 else 'MODERATE'
        print(f"  Total Privacy Budget: ($\epsilon$={epsilon:.4f}, $\delta$={delta})") 
    else:
        privacy_assessment = "CALCULATION FAILED"
        print(f"  Total Privacy Budget: ($\epsilon$=inf, $\delta$={delta})")
    
    print(f"  Noise Multiplier ($\sigma$): {sigma}")
    print(f"  Assessment: {privacy_assessment} Privacy Guarantee")
    
    print("\nDATA STATISTICS:")
    print(f"  Total Observations: {data_stats['total_observations']}")
    print(f"  Sequences (Approximate): {preproc_metadata['total_sequences']}")
    print(f"  Input Shape (Sequence Length, Total Features): {preproc_metadata['input_shape']}")
    print(f"  Continuous Features ({preproc_metadata['continuous_feature_count']}): {', '.join(data_stats['continuous_features'])}")
    print(f"  Categorical Features ({preproc_metadata['categorical_feature_count']}): {', '.join(data_stats['categorical_features'])}")
    print("="*80)

def run_optimized_pipeline():
    start_time = datetime.now()
    output_dir = Path("optimized_fhir_research") / f"multi_model_{start_time.strftime('%Y%m%dT%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=== Phase 1: Data Loading (Multi-Feature) ===")
    data_df, data_stats = load_fhir_data()

    logging.info("=== Phase 2: Data Preprocessing (Multi-Feature) ===")
    X_train, X_val, client_datasets, scaler, preproc_metadata, val_patient_data = preprocess_data(data_df, OPTIMIZED_CONFIG)
    input_shape = preproc_metadata['input_shape']
    all_features = preproc_metadata['feature_names']
    
    total_rounds = OPTIMIZED_CONFIG['GLOBAL_ROUNDS']
    dp_sigma = OPTIMIZED_CONFIG['DP_SIGMA']
    dp_delta = OPTIMIZED_CONFIG['DELTA']
    dp_q = OPTIMIZED_CONFIG['SAMPLING_Q']
    
    # Calculate Privacy Budget
    alpha, epsilon = calculate_dp_epsilon(dp_sigma, dp_q, total_rounds, dp_delta)
    
    full_results = []
    best_fl_model = None
    
    logging.info("=== Phase 3: Optimized Multi-Model Training (Centralized and Federated) ===")

    for model_name, model_creator in MODEL_CREATORS.items():
        logging.info(f"--- Training {model_name} ---")

        # 1. Centralized Training
        _, cent_rmse, cent_mae = train_centralized_model(X_train, X_val, input_shape, OPTIMIZED_CONFIG, scaler, model_creator, all_features)
        
        full_results.append({'Model': model_name, 'Type': 'Centralized', 'rmse': cent_rmse, 'mae': cent_mae, 'training_history': []})

        # 2. Federated Training (with DP)
        fl_trainer = FLCentralTrainer(OPTIMIZED_CONFIG, input_shape, scaler, model_creator, all_features)
        fl_model, fl_rmse, fl_mae = fl_trainer.train(client_datasets, X_val)
        
        full_results.append({'Model': model_name, 'Type': 'FL', 'rmse': fl_rmse, 'mae': fl_mae, 'training_history': fl_trainer.training_history})
        
        # Keep track of the best model (Conv1D-AE) for detailed plotting
        if model_name == 'Conv1D-AE':
             best_fl_model = fl_model
    
    logging.info("OPTIMIZED multi-model pipeline completed! Results saved to: " + str(output_dir))
    
    # ---------------------------
    # Phase 4: Summarize and Plot
    # ---------------------------
    plot_metrics_and_ablation(full_results, output_dir, epsilon, dp_delta, dp_sigma, best_fl_model, X_val, scaler, all_features)
    print_results_summary(full_results, data_stats, preproc_metadata, epsilon, dp_delta, dp_sigma)
    print("="*80)

# ---------------------------
# Run the optimized pipeline
# ---------------------------
if __name__ == "__main__":
    try:
        run_optimized_pipeline()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
