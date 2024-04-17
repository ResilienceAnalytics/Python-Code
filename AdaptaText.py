import os
import re
import string
import argparse
import datetime
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import MultiHeadAttention

"""
This script trains and evaluates a text classification model using TensorFlow and custom attention mechanisms. 
It is configurable via command-line arguments and supports various attention mechanisms, training configurations, 
and evaluation settings.

Usage:
    # To run the script with default parameters
    python AdaptaText.py

    # To run the script with a custom batch size and number of epochs
    python AdaptaText.py --batch-size 64 --epochs 20

    # To run the script with a custom attention mechanism and increased embedding dimension
    python AdaptaText.py --use-custom-attention --embed-dim 256 --attention-types 'cosine_similarity' 'l1_norm'

    # To run the script with standard attention, using a specific number of attention heads and dropout rate
    python AdaptaText.py --use-standard-attention --num-heads 4 --dropout-rate 0.3

    # To enable both standard and custom attention mechanisms
    python AdaptaText.py --use-standard-attention --use-custom-attention --attention-types 'scaled_dot_product' 'l2_norm' FIXME

Command-line Arguments:
    --use-standard-attention: Use TensorFlow's standard MultiHeadAttention.
    --use-custom-attention: Use a custom multi-head attention mechanism.
    --batch-size [int]: The number of samples in each batch (default: 32).
    --epochs [int]: The number of epochs to train the model (default: 10).
    --embedding-dim [int]: Dimension of the embedding layer (default: 16).
    --max-features [int]: Maximum number of words to consider in the vocabulary (default: 10000).
    --sequence-length [int]: Input sequence length after vectorization (default: 250).
    --embed-dim [int]: Embedding dimension for each attention head (default: 512).
    --num-heads [int]: Number of attention heads (default: 8).
    --dropout-rate [float]: Dropout rate for dropout layers (default: 0.2).
    --attention-types [list of str]: Types of attention mechanisms for each head (default: ['scaled_dot_product']).
    --seed [int]: Random seed for reproducibility (default: 15).
    --l1-regularizer [float]: L1 regularization coefficient (optional).
    --l2-regularizer [float]: L2 regularization coefficient (optional).
    --initial-lr [float]: Initial learning rate if using a learning rate scheduler (optional).
    --lr-decay-rate [float]: Decay rate for the learning rate scheduler (optional).
    --lr-decay-steps [int]: Decay steps for the learning rate scheduler (optional).
    --min-delta [float]: Minimum change in the monitored quantity to qualify as an improvement (default: 0.001).
    --patience [int]: Number of epochs with no improvement after which training will be stopped (default: 5).
    --verbose [int]: Verbosity mode (default: 1).
    --restore-best-weights: Whether to restore model weights from the epoch with the best value of the monitored quantity.
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a text classification model with flexible attention mechanisms.")
    # Model configuration
    parser.add_argument('--use-standard-attention', action='store_true',
                        help="Use TensorFlow's standard MultiHeadAttention instead of a custom implementation.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for training and validation data.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epochs to train the model.")
    parser.add_argument('--embedding-dim', type=int, default=16,
                        help="Dimension of the embedding layer.")
    parser.add_argument('--max-features', type=int, default=10000,
                        help="Maximum number of words to consider in the vocabulary.")
    parser.add_argument('--sequence-length', type=int, default=250,
                        help="Input sequence length after vectorization.")
    parser.add_argument('--embed-dim', type=int, default=512,
                        help="Embedding dimension for the attention layer.")
    parser.add_argument('--num-heads', type=int, default=8,
                        help="Number of attention heads.")
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help="Dropout rate for dropout layers.")
    parser.add_argument('--use-custom-attention', action='store_true', 
    			help="Use custom attention mechanisms for each head.")
    parser.add_argument('--attention-types', type=str, nargs='+', default=['scaled_dot_product'], 
    			help="List of attention types for each head in the custom attention layer.")
    parser.add_argument('--attention-type', type=str, default='scaled_dot_product',
                        help="Type of attention mechanism to use in the custom attention layer.")
    parser.add_argument('--use-dropout', action='store_true',
                        help="Toggle the use of dropout layers in the model.")
    parser.add_argument('--use-batchnorm', action='store_true',
                        help="Toggle the use of batch normalization in the model.")
                        
    parser.add_argument('--seed', type=int, default=15, help="Random seed for reproducibility.")

    # Regularization
    parser.add_argument('--l1-regularizer', type=float, default=None,
                        help="L1 regularization coefficient.")
    parser.add_argument('--l2-regularizer', type=float, default=None,
                        help="L2 regularization coefficient.")

    # Learning rate scheduler
    parser.add_argument('--use-lr-scheduler', action='store_true',
                        help="Use a learning rate scheduler for training.")
    parser.add_argument('--initial-lr', type=float, default=None,
                        help="Initial learning rate if using a scheduler.")
    parser.add_argument('--lr-decay-rate', type=float, default=None,
                        help="Decay rate for the learning rate scheduler.")
    parser.add_argument('--lr-decay-steps', type=int, default=None,
                        help="Decay steps for the learning rate scheduler.")

    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate for dropout layers.")
    parser.add_argument('--dense-units', type=int, default=1, help="Number of units in the Dense output layer.")
    parser.add_argument('--min-delta', type=float, default=0.001, help="Minimum change in the monitored quantity to qualify as an improvement.")
    parser.add_argument('--patience', type=int, default=5, help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument('--verbose', type=int, default=1, help="Verbosity mode.")
    parser.add_argument('--restore-best-weights', action='store_true', help="Whether to restore model weights from the epoch with the best value of the monitored quantity.")
    return parser.parse_args()

class CustomMultiheadAttention(tf.keras.layers.Layer):
    """
    A custom implementation of a multi-head attention mechanism that supports
    multiple types of attention within the same layer. This class allows for the
    specification of different attention types for each head, potentially combining
    them in innovative ways to enhance model performance on specific tasks.

    Attributes:
        embed_dim (int): The size of the embedding dimension for each attention head.
        num_heads (int): The number of attention heads in the layer.
        attention_types (list, optional): A list of attention mechanisms for each head.
            If not provided, all heads will use the 'scaled_dot_product' attention.
        dropout (float, optional): The dropout rate applied to the output of each attention head.

    Methods:
        call(inputs, mask=None): Processes the input data through all attention heads and
            concatenates their outputs.
        build(input_shape): Prepares the layer based on the input shape, typically setting
            up weights and biases.

    Example usage:
        # Define an attention layer with mixed types of attention mechanisms
        custom_attention = CustomMultiheadAttention(
            embed_dim=512, num_heads=4,
            attention_types=['scaled_dot_product', 'cosine_similarity', 'l1_norm', 'l2_norm'],
            dropout=0.1
        )
    """    
    def __init__(self, embed_dim, num_heads, attention_types=None, dropout=0.0):
        super(CustomMultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.heads = [AttentionHead(embed_dim, attention_type=attention_types[i] if attention_types and i < len(attention_types) else 'scaled_dot_product', dropout=dropout) for i in range(num_heads)]

    def call(self, inputs, mask=None):
        # Aggregate outputs from all heads
        outputs = [head(inputs, mask) for head in self.heads]
        # Simplified example of concatenating outputs
        concatenated = tf.concat([output for output in outputs], axis=-1)
        return concatenated

    def build(self, input_shape):
        super(CustomMultiheadAttention, self).build(input_shape)


    
class AttentionHead(tf.keras.layers.Layer):
    """
    This class implements a single attention head within a multi-head attention mechanism. It supports multiple types of attention calculations such as scaled dot product, cosine similarity, L1 and L2 norms, which allows it to be customized for different tasks and datasets. The attention type can be specified dynamically when creating an instance of this class.

    Attributes:
        embed_dim (int): Dimension of the embeddings for each of the queries, keys, and values.
        attention_type (str): Type of attention to apply. Supported types include 'scaled_dot_product', 'cosine_similarity', 'l1_norm', and 'l2_norm'.
        dropout (float): Dropout rate to apply to the outputs of the attention mechanism.

    Methods:
        call(inputs, mask=None): Processes the input using the specified attention mechanism and returns the result.
        apply_attention(q, k, v, mask): Applies the specified type of attention mechanism to the queries (q), keys (k), and values (v).

    Example:
        # Example of creating an AttentionHead with cosine similarity attention
        attention_head = AttentionHead(embed_dim=512, attention_type='cosine_similarity', dropout=0.1)
    """    
    def __init__(self, embed_dim, attention_type='scaled_dot_product', dropout=0.0):
        super(AttentionHead, self).__init__()
        self.attention_type = attention_type
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.Wq = tf.keras.layers.Dense(embed_dim, name="query")
        self.Wk = tf.keras.layers.Dense(embed_dim, name="key")
        self.Wv = tf.keras.layers.Dense(embed_dim, name="value")

    def call(self, inputs, mask=None):
        q = self.Wq(inputs)
        k = self.Wk(inputs)
        v = self.Wv(inputs)
        
        if self.dropout_layer.rate > 0:
            q = self.dropout_layer(q)
            k = self.dropout_layer(k)
            v = self.dropout_layer(v)

        return self.apply_attention(q, k, v, mask)

    def apply_attention(self, q, k, v, mask):
        """
        Applies the configured type of attention to the queries, keys, and values.

        Args:
            q (Tensor): Queries tensor.
            k (Tensor): Keys tensor.
            v (Tensor): Values tensor.
            mask (Tensor, optional): Mask tensor to exclude certain entries from attention calculation.

        Returns:
            Tensor: The resulting tensor after applying attention weights to the values.
        """
        # Implement different attention mechanisms based on self.attention_type
        if self.attention_type == 'cosine_similarity':
            return self.cosine_similarity_attention(q, k, v, mask)
        elif self.attention_type == 'l1_norm':
            return self.l1_norm_attention(q, k, v, mask)
        elif self.attention_type == 'l2_norm':
            return self.l2_norm_attention(q, k, v, mask)
        elif self.attention_type == 'scaled_dot_product':
            return self.dot_product_attention(q, k, v, mask)
        else:
            return self.dot_product_attention(q, k, v, mask)


    def cosine_similarity_attention(self, q, k, v, mask):
        q_normalized = tf.nn.l2_normalize(q, axis=-1)
        k_normalized = tf.nn.l2_normalize(k, axis=-1)
        matmul_qk = tf.matmul(q_normalized, k_normalized, transpose_b=True)
        if mask is not None:
            matmul_qk += (mask * -1e9)
        attention_weights = tf.nn.softmax(matmul_qk, axis=-1)
        return tf.matmul(attention_weights, v)

    def l1_norm_attention(self, q, k, v, mask):
        """
        Calculate attention weights based on L1 norm (Manhattan distance).
        
        Args:
            q (Tensor): Queries tensor.
            k (Tensor): Keys tensor.
            v (Tensor): Values tensor.
            mask (Tensor, optional): Mask tensor to exclude certain entries from attention.

        Returns:
            Tensor: The resulting tensor after applying attention weights to the values.
        """
        # Calculate the L1 distance
        q_expanded = tf.expand_dims(q, 2)
        k_expanded = tf.expand_dims(k, 1)
        l1_distance = tf.reduce_sum(tf.abs(q_expanded - k_expanded), axis=-1)

        if mask is not None:
            l1_distance += (mask * 1e9)

        attention_weights = tf.nn.softmax(-l1_distance, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output

    def l2_norm_attention(self, q, k, v, mask):
        """
        Calculate attention weights based on L2 norm (Euclidean distance).
        
        Args:
            q (Tensor): Queries tensor.
            k (Tensor): Keys tensor.
            v (Tensor): Values tensor.
            mask (Tensor, optional): Mask tensor to exclude certain entries from attention.

        Returns:
            Tensor: The resulting tensor after applying attention weights to the values.
        """
        # Calculate the L2 distance
        q_expanded = tf.expand_dims(q, 2)
        k_expanded = tf.expand_dims(k, 1)
        l2_distance = tf.sqrt(tf.reduce_sum(tf.square(q_expanded - k_expanded), axis=-1))

        if mask is not None:
            l2_distance += (mask * 1e9)

        attention_weights = tf.nn.softmax(-l2_distance, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output

    def differentialSum_attention(self, q, k, v, mask):
    # Calculate the difference between old and new embeddings
        delta_k = k[:, :, 1:] - k[:, :, :-1]
        delta_q = q[:, :, 1:] - q[:, :, :-1]

    # Print shapes of delta_k and delta_q for debugging
        print("Shape of delta_k:", delta_k.shape)
        print("Shape of delta_q:", delta_q.shape)

    # Calculate the product of differences
        delta_product = delta_k * q[:, :, :-1] + delta_q * k[:, :, 1:]

    # Print shape of delta_product for debugging
        print("Shape of delta_product:", delta_product.shape)

    # Calculate the sum of products for each dimension
        sum_product = tf.reduce_sum(delta_product, axis=-1, keepdims=True)

    # Print shape of sum_product for debugging
        print("Shape of sum_product:", sum_product.shape)

        if mask is not None:
            sum_product += (mask * -1e9)  # Apply mask
        # Print shape of sum_product after applying mask for debugging
            print("Shape of sum_product after mask:", sum_product.shape)

        attention_weights = tf.nn.softmax(sum_product, axis=1)
        attention_weights = tf.expand_dims(attention_weights, -1)  # Make sure it's [?, 250, 1]

    # Print shape of attention_weights after softmax and expand_dims for debugging
        print("Shape of attention_weights after softmax and expand_dims:", attention_weights.shape)

    # Correctly expand attention_weights to [?, 250, 256]
    # Ensure that the rank of 'attention_weights' and multiples are correct
        attention_weights = tf.expand_dims(attention_weights, 1)

    # Print shape of expanded attention_weights for debugging
        print("Shape of expanded attention_weights:", attention_weights.shape)

        output = tf.reduce_sum(attention_weights * v, axis=2)

    # Print shape of output for final debugging
        print("Shape of output:", output.shape)

        return output

    def differential_attention(self, q, k, v, mask):
        """
        Calculate attention weights based on the product of differences.

        Args:
            q (Tensor): Queries tensor containing old and new embeddings.
            k (Tensor): Keys tensor containing old and new embeddings.
            v (Tensor): Values tensor.
            mask (Tensor, optional): Mask tensor to exclude certain entries from attention.

        Returns:
            Tensor: The resulting tensor after applying attention weights to the values.
        """
        # Calculate the difference between old and new embeddings
        delta_k = k[:, :, 1:] - k[:, :, :-1]  # Assuming k contains embeddings in the order [old, new]
        delta_q = q[:, :, 1:] - q[:, :, :-1]  # Same for q

        # Calculate the product of differences
        delta_product = delta_k * q[:, :, :-1] + delta_q * k[:, :, 1:]

        # If a mask is provided, add a large negative value to masked positions
        if mask is not None:
            delta_product += (mask * -1e9)

        # Apply softmax directly to the product of differences to obtain attention weights
        attention_weights = tf.nn.softmax(delta_product, axis=-1)

        # Apply attention weights to the values
        output = tf.matmul(attention_weights, v)
        return output

    def dot_product_attention(self, q, k, v, mask):
        """
        Implementation of scaled dot product attention.

        Args:
            q (Tensor): Queries tensor.
            k (Tensor): Keys tensor.
            v (Tensor): Values tensor.
            mask (Tensor, optional): Mask tensor to exclude certain entries from attention.

        Returns:
            Tensor: The resulting tensor after applying attention weights to the values.
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # Dot product

        # Scaling by the square root of the depth of the keys
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # Adding a large negative value to masked positions

        # Softmax to obtain attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply attention weights to the values
        output = tf.matmul(attention_weights, v)
        return output



def custom_standardization(input_data):
    """Standardizes text by converting to lowercase and removing HTML tags and punctuation."""
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

def prepare_datasets(batch_size, seed, max_features, sequence_length):
    """
    Prepares training, validation, and test datasets for a text classification task using data from a specified directory.
    The function automates the process of dataset creation including loading data, splitting it into training and validation sets,
    and applying text vectorization.

    Args:
        batch_size (int): Number of samples per batch of computation.
        seed (int): Random seed for shuffling the data.
        max_features (int): Maximum number of words to keep based on frequency in the vocabulary.
        sequence_length (int): The maximum length of text sequences. Sequences longer than this will be truncated.

    Returns:
        tuple: Contains three tf.data.Dataset objects (train_ds, val_ds, test_ds).
    """
    # Path to the dataset directory
    dataset_path = '/home/hal/Documents/work/semantic/minitrasformers/evaluation/aclImdb/'

    # Create training and validation datasets from the training directory
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        dataset_path + 'train',
        batch_size=batch_size,
        validation_split=0.5,
        subset='training',
        seed=seed
    )
    
    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        dataset_path + 'train',
        batch_size=batch_size,
        validation_split=0.5,
        subset='validation',
        seed=seed
    )
    
    # Create a test dataset from the test directory
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        dataset_path + 'test',
        batch_size=batch_size
    )

    # Configure the text vectorization layer
    vectorize_layer = layers.TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    # Adapt the vectorization layer to the training data
    vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

    # Apply text vectorization to the datasets
    train_ds = raw_train_ds.map(lambda x, y: (vectorize_layer(x), y))
    val_ds = raw_val_ds.map(lambda x, y: (vectorize_layer(x), y))
    test_ds = raw_test_ds.map(lambda x, y: (vectorize_layer(x), y))

    return train_ds, val_ds, test_ds

def build_model(args):
    """
    Builds a neural network model configured with various types of attention mechanisms. The function supports using
    standard TensorFlow attention, custom multi-head attention, or a combination of both, depending on the flags set in
    the input arguments. The model is constructed as a sequential model using the Keras API.

    Args:
        args (Namespace): Configuration arguments including model dimensions, types of attention, and other hyperparameters.
            - args.use_standard_attention (bool): Flag to use TensorFlow's built-in MultiHeadAttention.
            - args.use_custom_attention (bool): Flag to use a custom multi-head attention mechanism.
            - args.num_heads (int): Number of attention heads.
            - args.embed_dim (int): Embedding dimension for each attention head.
            - args.attention_types (list of str): Specific attention mechanisms for each head in custom attention.
            - args.max_features (int): Number of words to consider from the input vocabulary.
            - args.embedding_dim (int): Dimensionality of the embedding layer.
            - args.dropout_rate (float): Dropout rate to apply after the attention mechanism.
            - args.dense_units (int): Number of neurons in the dense output layer of the model.

    Returns:
        tf.keras.Model: The constructed TensorFlow Keras model ready for training.

    Example usage:
        args = {
            'use_standard_attention': True,
            'use_custom_attention': False,
            'num_heads': 8,
            'embed_dim': 512,
            'attention_types': ['scaled_dot_product'],
            'max_features': 10000,
            'embedding_dim': 128,
            'dropout_rate': 0.1,
            'dense_units': 10
        }
        model = build_model(args)
    """
    if args.use_standard_attention and args.use_custom_attention:
        # Mixed mode: both standard and custom attention mechanisms
        attention_layers = []
        standard_attention = MultiHeadAttention(num_heads=args.num_heads, key_dim=args.embed_dim)
        custom_attention = CustomMultiheadAttention(embed_dim=args.embed_dim, num_heads=args.num_heads, attention_types=args.attention_types)
        attention_layers.append(standard_attention)
        attention_layers.append(custom_attention)
        attention_layer = layers.Concatenate()(attention_layers)
    elif args.use_standard_attention:
        attention_layer = MultiHeadAttention(num_heads=args.num_heads, key_dim=args.embed_dim)
    else:
        attention_layer = CustomMultiheadAttention(embed_dim=args.embed_dim, num_heads=args.num_heads, attention_types=args.attention_types)
    
    model = tf.keras.Sequential([
        layers.Embedding(args.max_features + 1, args.embedding_dim),
        attention_layer,
        layers.GlobalAveragePooling1D(),
        layers.Dropout(args.dropout_rate),
        layers.Dense(args.dense_units)
    ])
    return model


def train_and_evaluate(model, train_ds, val_ds, test_ds, callbacks):
    """
    Compiles, trains, and evaluates a TensorFlow Keras model using provided datasets and callbacks. This function
    handles the full lifecycle of training and validation including compiling the model, fitting it to the training data,
    and evaluating it on a test dataset.

    Args:
        model (tf.keras.Model): The TensorFlow Keras model to be trained and evaluated.
        train_ds (tf.data.Dataset): The dataset used for training the model.
        val_ds (tf.data.Dataset): The dataset used for validating the model during training.
        test_ds (tf.data.Dataset): The dataset used for final evaluation of the model to assess its performance.
        callbacks (list): A list of callbacks to use during training. For example, EarlyStopping, ModelCheckpoint, etc.

    Returns:
        tf.keras.Model: The trained and evaluated model.

    Example usage:
        model = build_model(args)
        train_ds, val_ds, test_ds = prepare_datasets(batch_size=32, seed=42, max_features=10000, sequence_length=250)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]
        trained_model = train_and_evaluate(model, train_ds, val_ds, test_ds, callbacks)
        print(f"Trained Model Evaluation: {trained_model.evaluate(test_ds)}")
    """
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=[tf.metrics.BinaryAccuracy()])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    loss, accuracy = model.evaluate(test_ds)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    return model

def setup_tensorboard():
    """
    Configures and returns a TensorBoard callback instance with a specific log directory based on the current date and time.
    This function sets up logging for TensorBoard, which allows you to monitor the training process with visualizations, histograms,
    and more, helping with the debugging and optimization of machine learning models.

    Returns:
        tf.keras.callbacks.TensorBoard: A configured TensorBoard callback instance with histogram frequency set to log at every epoch.

    Example usage:
        tensorboard_callback = setup_tensorboard()
        model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback])
    """
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return TensorBoard(log_dir=log_dir, histogram_freq=1)

if __name__ == "__main__":
    """
    Main execution block of the script: Initializes the model training and evaluation process.
    It parses command-line arguments, prepares datasets, builds the model, sets up callbacks including TensorBoard,
    and then trains and evaluates the model. Finally, it saves the trained model to a specified directory.

    Steps:
    1. Parse command-line arguments for model configuration.
    2. Prepare training, validation, and testing datasets.
    3. Build the model according to the specified configurations (e.g., attention mechanisms, dimensions).
    4. Setup the TensorBoard callback for monitoring the model's training.
    5. Define additional training callbacks such as EarlyStopping.
    6. Train and evaluate the model using the datasets and callbacks.
    7. Save the trained model to the filesystem.
    """
    args = parse_arguments()
    train_ds, val_ds, test_ds = prepare_datasets(
        batch_size=args.batch_size, 
        seed=args.seed, 
        max_features=args.max_features, 
        sequence_length=args.sequence_length
    )
    model = build_model(args)
    tensorboard_callback = setup_tensorboard()
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=args.min_delta, patience=args.patience, verbose=args.verbose, mode='min', restore_best_weights=args.restore_best_weights), tensorboard_callback]
    trained_model = train_and_evaluate(model, train_ds, val_ds, test_ds, callbacks)
    model.save('/home/hal/Documents/work/semantic/minitrasformers/evaluation/model')

