import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import logging
import math
from typing import Dict, Callable, Tuple, List
from tensorflow_privacy.privacy.analysis import rdp_accountant

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------
# CENTRALIZED TRAINING FUNCTION (Extracted from final_code_2.txt)
# --------------------------

def train_centralized_model(X_train: np.ndarray, X_val: np.ndarray, input_shape: tuple, config: Dict, scaler: Any, model_creator: Callable, all_features: List[str]) -> Tuple[tf.keras.Model, float, float]:
    """Trains a single model centrally for baseline comparison."""
    
    model = model_creator(input_shape, config)
    
    lr = config['TRAINING_CONFIG']['learning_rate']
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mse', 'mae'])
    
    batch_size = config['TRAINING_CONFIG']['batch_size']
    epochs = config['TRAINING_CONFIG']['global_rounds'] * 5 # Centralized training uses total epochs

    logging.info(f"Starting Centralized Training for {model.name}...")

    # Train (with Early Stopping if needed, but keeping it simple like the original code)
    model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        verbose=0 # Suppress verbose output
    )
    
    # Evaluate
    results = model.evaluate(X_val, X_val, verbose=0)
    rmse = math.sqrt(results[0])
    mae = results[2]
    
    logging.info(f"Centralized {model.name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    return model, rmse, mae

# --------------------------
# FEDERATED LEARNING TRAINER CLASS (Extracted from final_code_2.txt)
# --------------------------

class FLCentralTrainer:
    """
    Manages the central server logic for Differentially Private Federated Learning.
    
    Handles communication rounds, client model synchronization, FedAvg aggregation,
    and the application of DP noise.
    """
    def __init__(self, config: Dict, input_shape: tuple, scaler: Any, model_creator: Callable, all_features: List[str]):
        self.config = config
        self.input_shape = input_shape
        self.scaler = scaler
        self.all_features = all_features
        
        self.dp_enabled = config['DIFFERENTIAL_PRIVACY_CONFIG']['dp_enabled']
        self.dp_sigma = config['DIFFERENTIAL_PRIVACY_CONFIG']['dp_sigma']
        self.dp_clipping_norm = config['DIFFERENTIAL_PRIVACY_CONFIG']['dp_clipping_norm']
        self.n_clients = config['FL_CONFIG']['fraction_fit'] * config.get('N_CLIENTS', 3) # Use actual N_CLIENTS from data setup if available, otherwise default to 3
        
        # Initialize the global model
        self.global_model = model_creator(input_shape, config)
        self.global_model.compile(optimizer='sgd', loss='mse', metrics=['mse', 'mae']) # Optimizer is set on clients
        
        self.training_history = []

    def _create_client_model(self, model_creator: Callable) -> tf.keras.Model:
        """Creates a fresh client model instance."""
        model = model_creator(self.input_shape, self.config)
        
        # Determine optimizer based on DP setting
        lr = self.config['TRAINING_CONFIG']['learning_rate']
        
        if self.dp_enabled:
            # Use DP Adam Optimizer
            from tensorflow_privacy.keras.optimizers import DPMultiplicativeNoiseOptimiser
            optimizer = DPMultiplicativeNoiseOptimiser(
                l2_norm_clip=self.dp_clipping_norm,
                noise_multiplier=self.dp_sigma,
                # The original code used Adam in final_code_2.txt
                # We use the recommended DPMultiplicativeNoiseOptimiser wrapper
                optimizer=Adam(learning_rate=lr) 
            )
        else:
            optimizer = Adam(learning_rate=lr)
            
        model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])
        return model

    def train(self, client_datasets: Dict[int, np.ndarray], X_val: np.ndarray, model_creator: Callable) -> Tuple[tf.keras.Model, float, float]:
        """Runs the federated training loop."""
        
        global_rounds = self.config['TRAINING_CONFIG']['global_rounds']
        local_epochs = self.config['TRAINING_CONFIG']['local_epochs']
        batch_size = self.config['TRAINING_CONFIG']['batch_size']
        
        N_CLIENTS = len(client_datasets)
        
        logging.info(f"Starting FL Training ({self.global_model.name}). DP Enabled: {self.dp_enabled}")

        for round_num in range(1, global_rounds + 1):
            
            # 1. Distribute (Clients receive global weights)
            global_weights = self.global_model.get_weights()
            
            client_updates = []
            
            # 2. Local Training
            for client_id in range(N_CLIENTS):
                client_model = self._create_client_model(model_creator)
                client_model.set_weights(global_weights)
                
                X_train_client = client_datasets[client_id]
                
                # Train the client model
                client_model.fit(
                    X_train_client, X_train_client,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    verbose=0 # Suppress local training output
                )
                
                # Get the client's final weights
                client_updates.append(client_model.get_weights())
            
            # 3. Aggregate (FedAvg)
            new_global_weights = self._aggregate_weights(global_weights, client_updates, N_CLIENTS)
            
            # Update Global Model
            self.global_model.set_weights(new_global_weights)
            
            # 4. Evaluation and Logging
            results = self.global_model.evaluate(X_val, X_val, verbose=0)
            val_loss = results[0]
            val_rmse = math.sqrt(val_loss)
            val_mae = results[2]
            
            self.training_history.append(val_rmse)
            
            if round_num % 5 == 0 or round_num == 1:
                logging.info(f"Round {round_num}/{global_rounds} - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

        # Final Evaluation
        final_results = self.global_model.evaluate(X_val, X_val, verbose=0)
        final_rmse = math.sqrt(final_results[0])
        final_mae = final_results[2]
        
        return self.global_model, final_rmse, final_mae

    def _aggregate_weights(self, old_weights: List[np.ndarray], client_updates: List[List[np.ndarray]], n_clients: int) -> List[np.ndarray]:
        """Aggregates weights using FedAvg."""
        
        if not client_updates:
            return old_weights
        
        # Simple averaging (assuming equal client data sizes for simplicity)
        new_weights = [
            np.sum([client_update[i] for client_update in client_updates], axis=0) / n_clients
            for i in range(len(old_weights))
        ]
        
        return new_weights
