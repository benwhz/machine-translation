import keras
from keras import layers
import eng_fra_dataset

# dataset
vocab_size = eng_fra_dataset.vocab_size
sequence_length = eng_fra_dataset.sequence_length
batch_size = eng_fra_dataset.batch_size
train_ds = eng_fra_dataset.train_ds
val_ds = eng_fra_dataset.val_ds

embed_dim = 64
latent_dim = 128 

# GRU-based encoder.
source = keras.Input(shape=(None,), dtype="int64", name="english") 
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source) 
encoded_source = layers.Bidirectional(layers.GRU(latent_dim), merge_mode="sum")(x) 

# GRU-based decoder and the end-to-end model
past_target = keras.Input(shape=(None,), dtype="int64", name="french") 
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target) 
decoder_gru = layers.GRU(latent_dim, return_sequences=True)
x = decoder_gru(x, initial_state=encoded_source) 
x = layers.Dropout(0.5)(x)
target_next_step = layers.Dense(vocab_size, activation="softmax")(x) 
seq2seq_rnn = keras.Model([source, past_target], target_next_step) 
seq2seq_rnn.summary()

# compile and train.
seq2seq_rnn.compile(
 optimizer="rmsprop",
 loss="sparse_categorical_crossentropy",
 metrics=["accuracy"])
seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds)