import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
MAX_FEATURES = 10000
MAXLEN = 500
OOV_TOKEN = '<OOV>'

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
word_index = imdb.get_word_index()

# Improved print statements for data shapes
print("=== Data Shapes ===")
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing data shape: {x_test.shape}")
print(f"Testing labels shape: {y_test.shape}\n")

# Decode a sequence for demonstration
index_word = {v: k for k, v in word_index.items()}
decoded_sequence = " ".join(index_word.get(i - 3, OOV_TOKEN)
                            for i in x_train[0])
print("=== Decoded Sequence Example ===")
print(decoded_sequence + "\n")

# Pad sequences
x_train = pad_sequences(x_train, maxlen=MAXLEN)
x_test = pad_sequences(x_test, maxlen=MAXLEN)

# Model definition
model = Sequential([
    Embedding(MAX_FEATURES, 128, input_length=MAXLEN),
    SimpleRNN(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# Model summary
print("=== Model Summary ===")
model.summary()
print("\n")

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model
history = model.fit(x_train, y_train, batch_size=32, callbacks=[
                    early_stopping, model_checkpoint], validation_split=0.2, epochs=10)

# Evaluate the model
print("=== Model Evaluation ===")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}\n")
