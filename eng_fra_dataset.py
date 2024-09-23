'''

'''
import random
import string
import re
import tensorflow as tf
import keras

text_file = '../dataset/fra-eng/fra.txt'

# Parsing the data
with open(text_file, encoding='utf-8') as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, fra = line.split("\t")
    fra = "[start] " + fra + " [end]"
    text_pairs.append((eng, fra))
    
for _ in range(2):
    print(random.choice(text_pairs))
    
# split the sentence pairs into a training set, a validation set, and a test set.
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

# vectorizing the text data with TextVectorization
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 20
batch_size = 64

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

eng_vectorization = keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
fra_vectorization = keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

train_eng_texts = [pair[0] for pair in train_pairs]
train_fra_texts = [pair[1] for pair in train_pairs]
# computes a vocabulary of string terms from tokens in a dataset.
eng_vectorization.adapt(train_eng_texts)
fra_vectorization.adapt(train_fra_texts)
# check the vocabulary
print(eng_vectorization.get_vocabulary()[0:10])
print(fra_vectorization.get_vocabulary()[0:10])

# format dataset
def format_dataset(eng, fra):
    eng = eng_vectorization(eng)
    fra = fra_vectorization(fra)
    return (
        {
            "english": eng,
            "french": fra[:, :-1],  # without '[end]'
        },
        fra[:, 1:], # without '[start]'
    )

def make_dataset(pairs):
    eng_texts, fra_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    fra_texts = list(fra_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, fra_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["english"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["french"].shape}')
    print(f"targets.shape: {targets.shape}")

    #print(inputs["english"][0, :])
    #print(inputs["french"][0, :])
    #print(targets[0, :])
