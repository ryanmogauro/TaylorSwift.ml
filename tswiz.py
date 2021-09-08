from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import random
import sys


filename = "lyrics.txt"
#filename can be changed to train the model on specific styles/eras of her discography(lyrics.txt is her entire discography)
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
print(raw_text[0:1000])

raw_text = ''.join([x for x in raw_text if x.isdigit() == False])
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))


n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters in the text; corpus length: ", n_chars)
print("Total Vocab: ", n_vocab)



seq_length = 60  
step = 10   
sentences = []   
next_chars = []  
for i in range(0, n_chars - seq_length, step):  
    sentences.append(raw_text[i: i + seq_length])  
    next_chars.append(raw_text[i + seq_length])
n_patterns = len(sentences)    
print('Number of sequences:', n_patterns)


x = np.zeros((len(sentences), seq_length, n_vocab), dtype=np.bool)
y = np.zeros((len(sentences), n_vocab), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1
    
print(x.shape)
print(y.shape)

print(y[0:10])


model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, n_vocab)))
model.add(Dense(n_vocab, activation='softmax'))

optimizer = RMSprop(lr=0.007)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, n_vocab), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))

optimizer = RMSprop(lr=0.007)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

from keras.callbacks import ModelCheckpoint


filepath="saved_weights/saved_weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]


history = model.fit(x, y,
          batch_size=128,
          epochs=15, 
          callbacks=callbacks_list)

model.save('my_saved_weights_jungle_book_50epochs.h5')

from matplotlib import pyplot as plt


loss = history.history['loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1) 
    return np.argmax(probas)
def newLyrics():
  filename = "my_saved_weights_jungle_book_50epochs.h5"
  model.load_weights(filename)
  start_index = random.randint(0, n_chars - seq_length - 1)

  generated = ''
  sentence = raw_text[start_index: start_index + seq_length]
  generated += sentence

  print('----- Seed for our text prediction: "' + sentence + '"')


  for i in range(400):   # Number of characters including spaces
      x_pred = np.zeros((1, seq_length, n_vocab))
      for t, char in enumerate(sentence):
          x_pred[0, t, char_to_int[char]] = 1.

      preds = model.predict(x_pred, verbose=0)[0]
      next_index = sample(preds)
      next_char = int_to_char[next_index]

      generated += next_char
      sentence = sentence[1:] + next_char
  return generated
#print newLyrics() to generate novel lyrics in her style!





