import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle

model = keras.models.load_model("LunarAI.h5")
input_text = input("Texte à compléter: ")

with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=model.input_shape[1], padding='pre')
predicted_probs = model.predict(input_sequence, verbose=0)
num_words_to_generate = 25

generated_words = []
for _ in range(num_words_to_generate):
    predicted_index = np.argmax(predicted_probs, axis=1)[0]
    predicted_word = tokenizer.index_word.get(predicted_index, "")
    generated_words.append(predicted_word)
    input_sequence = np.append(input_sequence, [[predicted_index]], axis=1)
    input_sequence = input_sequence[:, 1:]
    predicted_probs = model.predict(input_sequence, verbose=0)

generated_text = " ".join(generated_words)

output_text = input_text + " " + generated_text
print(output_text)
