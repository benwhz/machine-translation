import os
import tensorflow as tf
import collections

reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        """Defined in :numref:`sec_text-sequence`"""
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

class English2FrenchData():
  def __init__(self, batch_size, seq_length = 10, num_train = 512, num_val = 128) -> None:
    self.root = '../dataset'
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.num_train = num_train
    self.num_val = num_val

    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
            self._read_file())
    
  def _read_file(self):
    with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
      return f.read()

  def _preprocess(self, text):
        """Defined in :numref:`sec_machine_translation`"""
        # Replace non-breaking space with space
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        # Insert space between words and punctuation marks
        no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
               for i, char in enumerate(text.lower())]
        return ''.join(out)

  def _tokenize(self, text, max_examples=None):
        """Defined in :numref:`sec_machine_translation`"""
        src, tgt = [], []
        for i, line in enumerate(text.split('\n')):
            if max_examples and i > max_examples: break
            parts = line.split('\t')
            if len(parts) == 2:
                # Skip empty tokens
                src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
                tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
        return src, tgt

  def _build_array(self, sentences, vocab, is_tgt=False):
            pad_or_trim = lambda seq, t: (
                seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
            sentences = [pad_or_trim(s, self.seq_length) for s in sentences]
            if is_tgt:
                sentences = [['<bos>'] + s for s in sentences]
            if vocab is None:
                vocab = Vocab(sentences, min_freq=2)
            array = tf.constant([vocab[s] for s in sentences])
            #valid_len = reduce_sum(astype(array != vocab['<pad>'], tf.int32), 1)
            return array, vocab

  def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
        """Defined in :numref:`subsec_loading-seq-fixed-len`"""
        src, tgt = self._tokenize(self._preprocess(raw_text),
                                  self.num_train + self.num_val)
        src_array, src_vocab  = self._build_array(src, src_vocab)
        tgt_array, tgt_vocab = self._build_array(tgt, tgt_vocab, True)
        return ((src_array, tgt_array[:,:-1], tgt_array[:,1:]),
                src_vocab, tgt_vocab)

  def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays

  def get_dataloader(self, train):
        """Defined in :numref:`subsec_loading-seq-fixed-len`"""
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        tensors = tuple(a[idx] for a in self.arrays)
        
        return tf.data.Dataset.from_tensor_slices(tensors)

if __name__ == '__main__':
  batch_size = 32

  data = English2FrenchData(batch_size)

  eng_voc = data.src_vocab
  fre_voc = data.tgt_vocab

  print(eng_voc.token_to_idx, fre_voc.token_to_idx)
  d_array = data.arrays

  print('src <pad> idx = ', eng_voc.token_to_idx['<pad>'])
  print('tgt <pad> idx = ', fre_voc.token_to_idx['<pad>'])

  dataloader = data.get_dataloader(train=True)
  for i, data in enumerate(dataloader):
    print(i, data)
    break
