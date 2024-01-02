import os
import re
import string
import argparse
import datetime
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import EarlyStopping

"""
| Attention Mechanism   | Validation Loss | Validation Accuracy | Test Loss | Test Accuracy |
|-----------------------|-----------------|---------------------|-----------|---------------|
| Differential          | 0.2904          | 0.8800              | 0.3087    | 0.8738        |
| DifferentialSum       | 0.2902          | 0.8794              | 0.3087    | 0.8737        |
| L2 Norm               | 0.2901          | 0.8794              | 0.3089    | 0.8734        |
| L1 Norm               | 0.2904          | 0.8800              | 0.3089    | 0.8737        |
| Cosine Similarity     | 0.2906          | 0.8798              | 0.3091    | 0.8730        |
| Scaled Dot Product    | 0.2902          | 0.8796              | 0.3093    | 0.8728        |

data:
https://www.tensorflow.org/tutorials/keras/text_classification?hl=fr#download_and_explore_the_imdb_dataset
"""

class CustomMultiheadAttention(tf.keras.layers.Layer):
    """
    Custom class for multi-head attention in TensorFlow. This layer implements a multi-head attention mechanism
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
        assert embed_dim % num_heads == 0
        self.attention_type = attention_type
        self.depth = embed_dim // num_heads
        self.Wq = tf.keras.layers.Dense(embed_dim)
        self.Wk = tf.keras.layers.Dense(embed_dim)
        self.Wv = tf.keras.layers.Dense(embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim)

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

        if self.attention_type == 'cosine_similarity':
            return self.cosine_similarity_attention(q, k, v, mask)
        elif self.attention_type == 'l1_norm':
            return self.l1_norm_attention(q, k, v, mask)
        elif self.attention_type == 'l2_norm':
            return self.l2_norm_attention(q, k, v, mask)
        elif self.attention_type == 'differentialSum':
            return self.differentialSum_attention(q, k, v, mask)
        elif self.attention_type == 'differential':
            return self.differential_attention(q, k, v, mask)
        return self.dot_product_attention(q, k, v, mask)

def custom_standardization(input_data):
        """Standardizes text by converting to lowercase and removing HTML tags and punctuation."""
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a text classification model with custom multi-head attention.")
    parser.add_argument('--attention-type', type=str, default='scaled_dot_product',
                        help="Type of attention mechanism ('scaled_dot_product', 'cosine_similarity', 'l1_norm', 'l2_norm', 'differentialSum', 'differential').")
    args = parser.parse_args()
    attention_type = args.attention_type

    # Dataset path configuration
    dataset_dir = os.path.join(os.path.dirname('//path/to/'))
    train_dir = os.path.join(dataset_dir, 'train')

    early_stopping = EarlyStopping(
    monitor='val_loss',   # Monitor the validation loss
    min_delta=0.001,      # Minimum change to consider as an improvement
    patience=5,           # Number of epochs with no improvement after which training will be stopped
    verbose=1,            # Display messages when training is stopped
    mode='min',           # 'Min' mode because the "loss" should decrease
    restore_best_weights=True  # Restore the model's weights to those of the best epoch
)

    # Prepare datasets
    batch_size = 32
    seed = 42
    max_features = 10000
    sequence_length = 250
    embedding_dim = 16
    epochs = 10
    callbacks=[early_stopping]

    raw_train_ds = tf.keras.utils.text_dataset_from_directory('/path/to/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)
    raw_val_ds = tf.keras.utils.text_dataset_from_directory('/path/to/val', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
    raw_test_ds = tf.keras.utils.text_dataset_from_directory('/path/to/test', batch_size=batch_size)

    vectorize_layer = layers.TextVectorization(standardize=custom_standardization, max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)
    vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

    train_ds = raw_train_ds.map(lambda x, y: (vectorize_layer(x), y))
    val_ds = raw_val_ds.map(lambda x, y: (vectorize_layer(x), y))
    test_ds = raw_test_ds.map(lambda x, y: (vectorize_layer(x), y))

    # Model construction
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        CustomMultiheadAttention(embed_dim=512, num_heads=8, attention_type=attention_type),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=[tf.metrics.BinaryAccuracy(threshold=0.0)])
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    loss, accuracy = model.evaluate(test_ds)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save path with date and time
    save_path = f'my_model_{current_time}'  # The path can be a directory
    model.save(save_path)
   
