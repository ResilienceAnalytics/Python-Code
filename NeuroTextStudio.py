import os
import re
import string
import argparse
import datetime
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import TensorBoard

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
    Custom  class for multi-head attention in TensorFlow. This layer implements a multi-head attention mechanism
    which allows the model to jointly attend to information from different representation subspaces.

    Args:
        embed_dim (int): Size of the embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        attention_type (str): Type of attention mechanism. Defaults to 'scaled_dot_product'.
    """
    def __init__(self, embed_dim, num_heads, attention_type='cosine_similarity', dropout=0.0):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.attention_type = attention_type
        self.depth = self.embed_dim // self.num_heads

    def build(self, input_shape):
        # Correctly use self.embed_dim to access the class member variable
        self.Wq = tf.keras.layers.Dense(self.embed_dim)
        self.Wk = tf.keras.layers.Dense(self.embed_dim)
        self.Wv = tf.keras.layers.Dense(self.embed_dim)
        self.dense = tf.keras.layers.Dense(self.embed_dim)
        super(CustomMultiheadAttention, self).build(input_shape)  # Ensure to call the super build method


    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def cosine_similarity_attention(self, q, k, v, mask):
        """
        Calculate attention weights based on cosine similarity.
        
        Args:
            q (Tensor): Queries tensor.
            v (Tensor): Values tensor.
            mask (Tensor, optional): Mask tensor to exclude certain entries from attention.

        Returns:
            Tensor: The resulting tensor after applying attention weights to the values.
        """
        # Normalize the query and key vectors
        q_normalized = tf.nn.l2_normalize(q, axis=-1)
        k_normalized = tf.nn.l2_normalize(k, axis=-1)

        # Compute the cosine similarity
        matmul_qk = tf.matmul(q_normalized, k_normalized, transpose_b=True)

        if mask is not None:
            matmul_qk += (mask * -1e9)

        attention_weights = tf.nn.softmax(matmul_qk, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output

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
        """
        Calculate attention weights based on the differential method by considering the difference between
        old and new embeddings.

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

        # Calculate the sum of products for each dimension
        sum_product = tf.reduce_sum(delta_product, axis=-1, keepdims=True)

        if mask is not None:
            sum_product += (mask * -1e9)

        attention_weights = tf.nn.softmax(sum_product, axis=-1)
        output = tf.matmul(attention_weights, v)
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
    # Chemin du dataset
    dataset_path = '/~/aclImdb/'

    # Création des datasets d'entraînement et de validation à partir du sous-dossier d'entraînement
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
    
    # Création du dataset de test à partir du sous-dossier de test
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        dataset_path + 'test',
        batch_size=batch_size
    )

    # Configuration de la couche de vectorisation du texte
    vectorize_layer = layers.TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    # Adaptation de la couche de vectorisation aux données d'entraînement
    vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

    # Vectorisation des textes dans les datasets
    train_ds = raw_train_ds.map(lambda x, y: (vectorize_layer(x), y))
    val_ds = raw_val_ds.map(lambda x, y: (vectorize_layer(x), y))
    test_ds = raw_test_ds.map(lambda x, y: (vectorize_layer(x), y))

    return train_ds, val_ds, test_ds

def build_model(attention_type, max_features, embedding_dim):
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        CustomMultiheadAttention(embed_dim=args.embedding_dim, num_heads=args.num_heads, attention_type=attention_type),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    return model

def train_and_evaluate(model, train_ds, val_ds, test_ds, callbacks):
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=[tf.metrics.BinaryAccuracy()])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    loss, accuracy = model.evaluate(test_ds)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    return model

def setup_tensorboard():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return TensorBoard(log_dir=log_dir, histogram_freq=1)

if __name__ == "__main__":
    args = parse_arguments()
    callbacks = []
    train_ds, val_ds, test_ds = prepare_datasets(
        batch_size=args.batch_size, 
        seed=args.seed, 
        max_features=args.max_features, 
        sequence_length=args.sequence_length
    )
    model = build_model(attention_type=args.attention_type, max_features=args.max_features, embedding_dim=args.embedding_dim)
    tensorboard_callback = setup_tensorboard()
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=args.min_delta, patience=args.patience, verbose=args.verbose, mode='min', restore_best_weights=args.restore_best_weights), tensorboard_callback]
    trained_model = train_and_evaluate(model, train_ds, val_ds, test_ds, callbacks)
    model.save('/~/aclImdb/model')
