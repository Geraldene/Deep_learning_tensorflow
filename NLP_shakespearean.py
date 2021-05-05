from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np

data = open('H:/Tensorflow_Coursera/sonnets.txt').read()

corpus = data.lower().split('\n')

tokenizer = Tokenizer(oov_token='oov')
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequences = token_list[:i+1]
        input_sequences.append(n_gram_sequences)

#pad sequences
max_sequence_len = max([len(x) for x in input_sequences])

padded_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding= 'pre'))

#create predictor and label
predictor, label = padded_sequences[:,:-1], padded_sequences[:,-1]

label = ku.to_categorical(label, num_classes=total_words)

#Deep learning model

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2=1e-4)))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
history = model.fit(padded_sequences, label, epochs = 50, verbose =1 )


seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding = 'pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)

