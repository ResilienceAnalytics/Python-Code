import os
import re
import string
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses

class CustomMultiheadAttention(tf.keras.layers.Layer):
    """
    Custom class for multi-head attention in TensorFlow. This layer implements a multi-head attention mechanism
    which allows the model to jointly attend to information from different representation subspaces.

    Args:
        embed_dim (int): Size of the embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0

        self.depth = embed_dim // num_heads
        self.Wq = tf.keras.layers.Dense(embed_dim)
        self.Wk = tf.keras.layers.Dense(embed_dim)
        self.Wv = tf.keras.layers.Dense(embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        """Splits the last dimension of the tensor into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask=None):
        """Process the input through the multi-head attention mechanism."""
        batch_size = tf.shape(x)[0]
        q = k = v = x

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Attention calculation
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_output = tf.reshape(output, (batch_size, -1, self.embed_dim))
        return self.dense(concat_output)

# Dataset path configuration
dataset_dir = os.path.join(os.path.dirname('/home/hal/Documents/work/Publication/minitrasformers/evaluation/aclImdb/'))
train_dir = os.path.join(dataset_dir, 'train')

# Prepare datasets
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    '/home/hal/Documents/work/Publication/minitrasformers/evaluation/aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    '/home/hal/Documents/work/Publication/minitrasformers/evaluation/aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    '/home/hal/Documents/work/Publication/minitrasformers/evaluation/aclImdb/test', 
    batch_size=batch_size)

# Text preprocessing
def custom_standardization(input_data):
    """Custom text standardization function to lowercase and remove HTML tags and punctuation."""
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

# Map datasets
train_ds = raw_train_ds.map(lambda x, y: (vectorize_layer(x), y))
val_ds = raw_val_ds.map(lambda x, y: (vectorize_layer(x), y))
test_ds = raw_test_ds.map(lambda x, y: (vectorize_layer(x), y))

# Model construction
embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  CustomMultiheadAttention(embed_dim=512, num_heads=8),  # Major difference
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Training and evaluation
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Training the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# Model evaluation
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Data for graphs
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
epochs = range(1, len(acc) + 1)

# Loss graph
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy graph
plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
