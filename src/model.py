import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Adam
from typing import Dict, Callable

def create_conv1d_ae(input_shape: tuple, config: Dict) -> Model:
    """Creates the 1D Convolutional Autoencoder."""
    latent_dim = config['MODEL_CONFIG']['latent_dim']
    conv_layers = config['MODEL_CONFIG']['conv_layers']
    
    input_seq = layers.Input(shape=input_shape)
    x = input_seq

    # Encoder
    for i in range(conv_layers):
        x = layers.Conv1D(filters=32 * (2**i), kernel_size=3, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    
    # Bottleneck
    x = layers.Flatten()(x)
    bottleneck = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(tf.shape(x)[-1], activation='relu')(bottleneck)
    
    # Reshape for Decoder
    encoder_output_shape = tf.shape(input_seq)[1] // (2**conv_layers)
    x = layers.Reshape((encoder_output_shape, 32 * (2**(conv_layers-1))))(x)

    # Decoder
    for i in range(conv_layers - 1, -1, -1):
        filters = 32 * (2**i)
        x = layers.Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
        # UpSampling to reverse MaxPool. Use strides=2 or UpSampling1D
        x = layers.UpSampling1D(size=2)(x) 
    
    # Final Layer
    # Crop is needed because UpSampling can lead to slightly larger sequence length
    crop_size = tf.shape(x)[1] - input_shape[0]
    if crop_size > 0:
        x = layers.Cropping1D(cropping=(0, crop_size))(x)
        
    output_seq = layers.Conv1D(filters=input_shape[-1], kernel_size=3, padding='same', activation='linear')(x)
    
    return Model(inputs=input_seq, outputs=output_seq, name="Conv1D-AE")

def create_bilstm_ae(input_shape: tuple, config: Dict) -> Model:
    """Creates the Bidirectional LSTM Autoencoder."""
    latent_dim = config['MODEL_CONFIG']['latent_dim']
    lstm_units = config['MODEL_CONFIG']['lstm_units']
    
    input_seq = layers.Input(shape=input_shape)
    
    # Encoder
    encoded = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, name="bi_lstm_1"))(input_seq)
    encoded = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=False, name="bi_lstm_2"))(encoded)
    
    # Decoder
    # Repeat vector expands the latent vector back to the sequence length
    decoded = layers.RepeatVector(input_shape[0])(encoded)
    decoded = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, name="bi_lstm_3"))(decoded)
    
    # Final dense layer to match feature dimension
    output_seq = layers.TimeDistributed(layers.Dense(input_shape[-1], activation='linear'))(decoded)
    
    return Model(inputs=input_seq, outputs=output_seq, name="BiLSTM-AE")

def create_transformer_ae(input_shape: tuple, config: Dict) -> Model:
    """Creates the Transformer-based Autoencoder."""
    latent_dim = config['MODEL_CONFIG']['latent_dim']
    transformer_units = config['MODEL_CONFIG']['transformer_units']
    transformer_heads = config['MODEL_CONFIG']['transformer_heads']
    
    input_seq = layers.Input(shape=input_shape)
    
    # --- Transformer Encoder Block ---
    # Multi-Head Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=transformer_heads, key_dim=input_shape[-1] // transformer_heads
    )(input_seq, input_seq)
    
    x = layers.Dropout(0.1)(attention_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + input_seq)

    # Feed Forward Network (FFN)
    ffn_output = layers.Conv1D(filters=transformer_units, kernel_size=1, activation="relu")(x)
    ffn_output = layers.Conv1D(filters=input_shape[-1], kernel_size=1)(ffn_output)
    
    encoded = layers.LayerNormalization(epsilon=1e-6)(ffn_output + x)
    
    # --- Bottleneck ---
    # Reduce the sequence to the latent dimension (e.g., via mean pooling or a final dense layer)
    bottleneck_input = layers.Flatten()(encoded)
    bottleneck = layers.Dense(latent_dim, activation='relu')(bottleneck_input)
    
    # --- Transformer Decoder Block (Mirroring the structure) ---
    # Repeat the latent vector back to sequence length
    decoded_input = layers.Dense(input_shape[0] * input_shape[-1])(bottleneck)
    decoded_input = layers.Reshape(input_shape)(decoded_input)
    
    # Decoder Attention
    attention_output_dec = layers.MultiHeadAttention(
        num_heads=transformer_heads, key_dim=input_shape[-1] // transformer_heads
    )(decoded_input, decoded_input)
    
    y = layers.Dropout(0.1)(attention_output_dec)
    y = layers.LayerNormalization(epsilon=1e-6)(y + decoded_input)
    
    # Decoder FFN
    ffn_output_dec = layers.Conv1D(filters=transformer_units, kernel_size=1, activation="relu")(y)
    ffn_output_dec = layers.Conv1D(filters=input_shape[-1], kernel_size=1)(ffn_output_dec)
    
    decoded = layers.LayerNormalization(epsilon=1e-6)(ffn_output_dec + y)
    
    # Final output layer
    output_seq = layers.Conv1D(filters=input_shape[-1], kernel_size=1, activation='linear')(decoded)
    
    return Model(inputs=input_seq, outputs=output_seq, name="Transformer-AE")
