import tensorflow as tf
from tensorflow import keras
import pickle

text = open("Train.txt", "r").read()

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

input_sequences = tf.convert_to_tensor(input_sequences)
x_data = input_sequences[:, :-1]
y_data = tf.keras.utils.to_categorical(input_sequences[:, -1], num_classes=total_words)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(150, return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GRU(100))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_data, y_data, epochs=100, verbose=1)

model.save("LunarAI.h5")
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)