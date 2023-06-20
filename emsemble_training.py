import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Attention, GlobalAveragePooling1D, Dropout, LayerNormalization, MultiHeadAttention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from gensim.models import KeyedVectors

# Load pretrained word embeddings
embedding_file = 'path/to/pretrained/embeddings/file'  # Replace with the actual path
word_vectors = KeyedVectors.load_word2vec_format(embedding_file, binary=False)

# Example data
texts = [
    "The cat sat on the mat",
    "The dog played in the garden",
    "The bird sang a beautiful song"
]
summaries = [
    "The cat on the mat",
    "The dog in the garden",
    "The bird sang"
]

# Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts + summaries)

vocab_size = len(tokenizer.word_index) + 1

# Tokenize sequences
text_sequences = tokenizer.texts_to_sequences(texts)
summary_sequences = tokenizer.texts_to_sequences(summaries)

# Pad sequences
max_text_len = max(len(seq) for seq in text_sequences)
max_summary_len = max(len(seq) for seq in summary_sequences)

text_sequences = pad_sequences(text_sequences, padding='post', maxlen=max_text_len)
summary_sequences = pad_sequences(summary_sequences, padding='post', maxlen=max_summary_len)

x = text_sequences
y = summary_sequences

# Prepare inputs and targets for training
encoder_inputs = x
decoder_inputs = y[:, :-1]
decoder_targets = y[:, 1:]

# Create embedding matrix
embedding_dim = 300  # Embedding dimension corresponding to the pretrained embeddings
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in tokenizer.word_index.items():
    if word in word_vectors:
        embedding_matrix[idx] = word_vectors[word]

# Define model architecture
hidden_units = 256
num_attention_heads = 4  # Adjust the number of attention heads
num_encoder_layers = 2  # Set the number of encoder layers
num_decoder_layers = 2  # Set the number of decoder layers
dropout_rate = 0.2
l2_regularization = 0.001

encoder_input = Input(shape=(None,))
decoder_input = Input(shape=(None,))
encoder_embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)(encoder_input)
decoder_embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)(decoder_input)

encoder = encoder_embedding

# Add multiple encoder layers
for _ in range(num_encoder_layers):
    encoder = LSTM(hidden_units, return_sequences=True)(encoder)
    encoder = Dropout(dropout_rate)(encoder)

decoder = decoder_embedding

# Add multiple decoder layers
for _ in range(num_decoder_layers):
    attention = MultiHeadAttention(num_attention_heads, key_dim=hidden_units // num_attention_heads)(decoder, encoder, return_attention_scores=False)
    attention = Dropout(dropout_rate)(attention)
    attention = LayerNormalization(epsilon=1e-6)(attention)
    decoder = Concatenate()([decoder, attention])
    decoder = LSTM(hidden_units, return_sequences=True)(decoder)
    decoder = Dropout(dropout_rate)(decoder)
    decoder = LayerNormalization(epsilon=1e-6)(decoder)

decoder_outputs = GlobalAveragePooling1D()(decoder)
decoder_outputs = Dropout(dropout_rate)(decoder_outputs)
decoder_outputs = Dense(vocab_size, activation='softmax', kernel_regularizer=l2(l2_regularization))(decoder_outputs)

model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_outputs])

# Compile the model
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True))

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model on existing data
model.fit([encoder_inputs, decoder_inputs], decoder_targets, epochs=10, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Load the best model
model = tf.keras.models.load_model('best_model.h5')

# Generate summaries
def generate_summary(text):
    text_seq = tokenizer.texts_to_sequences([text])
    text_seq = pad_sequences(text_seq, padding='post', maxlen=max_text_len)
    predicted = model.predict([text_seq, np.zeros((1, max_summary_len - 1))])
    predicted = np.argmax(predicted, axis=-1)
    summary = ' '.join(tokenizer.index_word[idx] for idx in predicted[0] if idx > 0)
    return summary

# Example of training on new data
new_texts = [
    "The cat is sleeping",
    "The dog is running",
    "The bird is chirping"
]
new_summaries = [
    "The cat is asleep",
    "The dog is active",
    "The bird is singing"
]

new_x = tokenizer.texts_to_sequences(new_texts)
new_y = tokenizer.texts_to_sequences(new_summaries)

new_x = pad_sequences(new_x, padding='post', maxlen=max_text_len)
new_y = pad_sequences(new_y, padding='post', maxlen=max_summary_len)

# Train a new model on new data
new_encoder_inputs = new_x
new_decoder_inputs = new_y[:, :-1]
new_decoder_targets = new_y[:, 1:]

new_model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_outputs])
new_model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True))
new_model.fit([new_encoder_inputs, new_decoder_inputs], new_decoder_targets, epochs=10, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Load the best model trained on new data
new_model = tf.keras.models.load_model('best_model.h5')

# Create an ensemble of the existing model and the new model
ensemble_model = tf.keras.models.Sequential()
ensemble_model.add(model)
ensemble_model.add(new_model)

# Generate summaries using the ensemble model
def generate_summary_ensemble(text):
    text_seq = tokenizer.texts_to_sequences([text])
    text_seq = pad_sequences(text_seq, padding='post', maxlen=max_text_len)
    predicted = ensemble_model.predict([text_seq, np.zeros((1, max_summary_len - 1))])
    predicted = np.argmax(predicted, axis=-1)
    summary = ' '.join(tokenizer.index_word[idx] for idx in predicted[0] if idx > 0)
    return summary
