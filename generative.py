import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample training text
data = """
Artificial intelligence is transforming the world.
Machine learning helps computers learn from data.
Deep learning is a powerful subset of machine learning.
AI is used in healthcare, finance, education, and robotics.
"""

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.split("\n"))

total_words = len(tokenizer.word_index) + 1

# Create sequences
input_sequences = []
for line in data.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)

# Pad sequences
max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Split predictors and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build LSTM model
model = Sequential([
    Embedding(total_words, 64, input_length=max_sequence_len-1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=100, verbose=1)

# Reverse dictionary for word lookup
reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}

# Text generation function
def generate_text(seed_text, next_words=20):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
        output_word = reverse_word_index.get(predicted, "")
        seed_text += " " + output_word

    return seed_text

# Test generation
print(generate_text("artificial intelligence", 15))
