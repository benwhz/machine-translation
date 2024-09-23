import math
import tensorflow as tf
import keras
import keras.api.layers as layers
from dataset_tf import English2FrenchData
import time

class EmbeddingLayer(layers.Layer):
    def __init__(self, vocab_size, d_model):
        '''
        Args:
            vacab_size:     size of vocabulary
            d_model:        dimension of embeddings
        '''
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.emb_table = layers.Embedding(vocab_size, d_model)

    def call(self, x):
        '''
        Args:
            x:      input tensor, shape = (batch_size, seq_length)
        
        Return: 
            embedding vectors, shape = (batch_size, seq_length, d_model)
        '''
        return self.emb_table(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(layers.Layer):
  def __init__(self, d_model: int, drop_rate: float, max_length: int = 5000):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    super().__init__(trainable=False)
    
    self.dropout = layers.Dropout(drop_rate)
    
    #self.pe = tf.zeros((max_length, d_model))    

    k = tf.expand_dims(tf.range(0, max_length, dtype=tf.float32), axis=1)  
    
    div_term = tf.exp(                                 
            tf.range(0, d_model, 2, dtype=tf.float32) * -(math.log(10000.0) / d_model)
    )
    
    self.pe = tf.reshape(
        tf.concat([tf.sin(k * div_term)[...,tf.newaxis], tf.cos(k * div_term)[...,tf.newaxis]], axis=-1), 
        [tf.shape(k)[0],-1])
    
    self.pe = tf.expand_dims(self.pe, axis=0)

    # print(self.pe.shape)

  def call(self, x):
    """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)
    
    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    # add positional encoding to the embeddings
    x = x + self.pe[:, : x.shape[1]]

    # perform dropout
    return self.dropout(x)

class MultiHeadAttention(layers.Layer):
  def __init__(self, model_size, headers, drop_rate):
      super().__init__()           
      self.W_q = layers.Dense(model_size)
      self.W_k = layers.Dense(model_size)
      self.W_v = layers.Dense(model_size)
      self.headers = headers
      self.query_size = int(model_size/headers)
      self.W_o = layers.Dense(model_size)
      self.permute = keras.layers.Permute((2, 1))
      self.dropout = layers.Dropout(drop_rate)

  def _seperate_by_header(self, v):
      batch_size, seq_length, model_size = v.shape
      #print('++++', batch_size, seq_length, model_size, self.headers, self.query_size)
      v = tf.reshape(v, [-1, seq_length, self.headers, self.query_size])
      #print(v.shape)
      v = tf.transpose(v, perm=[0, 2, 1, 3]) 
      #v = keras.layers.Permute((2, 1))(v)
      #print(v.shape)
      return v

  def _aggregate_by_head(self, v):
      _, _headers, seq_length, _dim = v.shape

      v = tf.transpose(v, perm=[0, 2, 1, 3]) 

      v = tf.reshape(v, [-1, seq_length, _headers*_dim])
      return v
  
  def call(self, query, keys, mask=None):
      #print(query.shape)
      _query = self.W_q(query)
      _query = self._seperate_by_header(_query)
      _keys = self._seperate_by_header(self.W_k(keys))
      _values = self._seperate_by_header(self.W_v(keys))

      #print(_query.shape, _keys.shape)
      _keys_t = tf.transpose(_keys, perm=[0, 1, 3, 2]) 
      #print(_query.shape, _keys_t.shape)
      _scores = tf.matmul(_query, _keys_t) / math.sqrt(self.query_size)
      #print(_scores.shape)

      # mask shape = (batch_size, 1, 1, seq_length) or (batch_size, 1, seq_length, seq_length)
      # print('attention mask shape = ', mask.shape, _scores.shape)
      if mask is not None:
        tf.where(mask == False, tf.fill(tf.shape(_scores), -1e9), _scores)

      _weights = self.dropout(tf.nn.softmax(_scores, axis = -1))

      context = tf.matmul(_weights, _values)
      #print(context.shape)

      context = self._aggregate_by_head(context)
      context = self.W_o(context)

      return context

  def build(self, input_shape):
     pass

class FeedForwardLayer(layers.Layer):
  def __init__(self, d_model, drop_rate, d_ffn = 2048) -> None:
      super().__init__()
      self.liner1 = layers.Dense(d_ffn)
      self.relu = layers.ReLU()
      self.liner2 = layers.Dense(d_model)
      self.dropout = layers.Dropout(drop_rate)

  def call(self, x):
    return self.liner2(self.dropout(self.relu(self.liner1(x))))     

class EncodeBlock(layers.Layer):
    def __init__(self, d_model, header, drop_rate) -> None:
      super().__init__()         
      self.attention = MultiHeadAttention(d_model, header, drop_rate)
      self.norm1 = layers.LayerNormalization()
      self.ffn = FeedForwardLayer(d_model, drop_rate)
      self.norm2 = layers.LayerNormalization()
      self.dropout = layers.Dropout(drop_rate)

    def call(self, x, _mask):
       x = self.norm1(x + self.dropout(self.attention(x, x, _mask)))
       x = self.norm2(x + self.dropout(self.ffn(x)))
       return x

class DecodeBlock(layers.Layer):
    def __init__(self, d_model, header, drop_rate) -> None:
      super().__init__()         
      self.attention1 = MultiHeadAttention(d_model, header, drop_rate)
      self.norm1 = layers.LayerNormalization()
      self.attention2 = MultiHeadAttention(d_model, header, drop_rate)
      self.norm2 = layers.LayerNormalization()
      self.ffn = FeedForwardLayer(d_model, drop_rate)
      self.norm3 = layers.LayerNormalization()
      self.dropout = layers.Dropout(drop_rate)

    def call(self, x, context, src_mask, tgt_mask):
       x = self.norm1(x + self.dropout(self.attention1(x, x, tgt_mask)))
       x = self.norm2(x + self.dropout(self.attention2(x, context, src_mask)))
       x = self.norm3(x + self.dropout(self.ffn(x)))
       return x

class Encoder(layers.Layer):
   def __init__(self, d_model, header, n_block, drop_rate: float = 0.1) -> None:
      super().__init__()
      self.blocks = ([EncodeBlock(d_model, header, drop_rate) for n in range(n_block)])
      self.dropout = layers.Dropout(drop_rate)

   def call(self, x, _mask = None):
      for encode in self.blocks:
         x = encode(x, _mask)

      return self.dropout(x)

class Decoder(layers.Layer):
   def __init__(self, d_model, header, n_block, drop_rate: float = 0.1) -> None:
      super().__init__()
      self.blocks = ([DecodeBlock(d_model, header, drop_rate) for n in range(n_block)])
      self.dropout = layers.Dropout(drop_rate)

   def call(self, x, context, src_mask = None, tgt_mask = None):
      for decode in self.blocks:
         x = decode(x, context, src_mask, tgt_mask)

      return self.dropout(x)

class EndLayer(layers.Layer):
   def __init__(self, vocab_size):
      super().__init__()
      self.liner = layers.Dense(vocab_size)
      self.softmax = layers.Softmax()

   def call(self, x):
      x = (self.liner(x))
      return x
   
class EncoderMaskLayer(layers.Layer):
  def __init__(self, pad_idx):
     super().__init__()
     self.src_pad_idx = pad_idx
     
  def call(self, src):
    # (batch_size, seq_length)
    src_mask = (src != self.src_pad_idx)
    # (batch_size, 1, seq_length)
    src_mask = tf.expand_dims(src_mask, 1)
    # (batch_size, seq_length, seq_length)
    src_mask = src_mask & tf.transpose(src_mask, perm = (0, 2, 1))
    # (batch_size, 1, seq_length, seq_length)
    return tf.expand_dims(src_mask, axis=1)

class DecoderMaskLayer(layers.Layer):
  def __init__(self, pad_idx):
     super().__init__()
     self.trg_pad_idx = pad_idx
     
  def call(self, trg):
    seq_length = trg.shape[1]
    trg_mask = (trg != self.trg_pad_idx)
    trg_mask = tf.expand_dims(tf.expand_dims(trg_mask, 1), axis=1)
    ones_mask = tf.ones((seq_length, seq_length))
    ones_mask = (tf.linalg.band_part(ones_mask, -1, 0) == 1)
    # broadcasting bitwise and operation: (batch_size, 1, 1, seq_length) & (seq_length, seq_length) => ((batch_size, 1, seq_length, seq_length)
    trg_mask = trg_mask & ones_mask
    return trg_mask
     
class Transformer(keras.Model):
  def __init__(self, *args, **argv):
      super().__init__()
      src_vocab_size, tgt_vcab_size, d_model, header, stack_size = args
      self.enc_embedding = EmbeddingLayer(src_vocab_size, d_model)
      self.dec_embedding = EmbeddingLayer(tgt_vcab_size, d_model)
      
      self.pos_encoding = PositionalEncoding(d_model, 0.1)
      self.encoder = Encoder(d_model, header, stack_size, 0.1)
      self.decoder = Decoder(d_model, header, stack_size, 0.1)
      
      self.endl = EndLayer(tgt_vcab_size)

      self.src_pad_idx = 4461
      self.trg_pad_idx = 3110
      if 'src_pad_idx' in argv.keys():
        self.src_pad_idx = argv['src_pad_idx']
      if 'tgt_pad_idx' in argv.keys():
        self.trg_pad_idx = argv['tgt_pad_idx']

      self.enc_mask = EncoderMaskLayer(self.src_pad_idx)
      self.dec_mask = DecoderMaskLayer(self.trg_pad_idx)
      
  def build(self, input_shape):
      '''
      self.w = self.add_weight(
        shape=(input_shape[-1], self.units),
        initializer="random_normal",
        trainable=True,
      )

      self.b = self.add_weight(
        shape=(self.units,), initializer="random_normal", trainable=True
      )
      '''

  def call(self, inputs):
    #print(type(inputs), len(inputs), inputs[0][0], inputs[0][1])
    #exit()
    enc_inputs, dec_inputs = inputs[0], inputs[1]
    
    _enc_mask = self.enc_mask(enc_inputs)
    _dec_mask = self.dec_mask(dec_inputs)

    enc_x = self.enc_embedding(enc_inputs)
    enc_x = self.pos_encoding(enc_x)
    dec_x = self.dec_embedding(dec_inputs)
    dec_x = self.pos_encoding(dec_x)
      
    c = self.encoder(enc_x, _enc_mask)
    x = self.decoder(dec_x, c, _dec_mask, _enc_mask)
    x = self.endl(x)
    return x
  
  def build_graph(self, seq_length):
    enc_x = tf.keras.layers.Input(shape=(seq_length,))
    dec_x = tf.keras.layers.Input(shape=(seq_length,))
    
    return tf.keras.Model(inputs=(enc_x, dec_x), 
                          outputs=self.call((enc_x, dec_x)))  

  # error mask
  def make_src_mask(self, src):
    src_mask = (src != self.src_pad_idx)
    return tf.expand_dims(tf.expand_dims(src_mask, 1), axis=1)

  def make_trg_mask(self, trg):
    seq_length = trg.shape[1]
    trg_mask = (trg != self.trg_pad_idx)
    trg_mask = tf.expand_dims(tf.expand_dims(trg_mask, 1), axis=1)
    
    ones_mask = tf.ones((seq_length, seq_length))
    ones_mask = (tf.linalg.band_part(ones_mask, -1, 0) == 1)
    # broadcasting bitwise and operation: (batch_size, 1, 1, seq_length) & (seq_length, seq_length) => ((batch_size, 1, seq_length, seq_length)
    trg_mask = trg_mask & ones_mask
    return trg_mask

def mask_sample():
    data = tf.ones((3, 4, 5, 5))
    mask = tf.constant([[True, True, True, True, False]])
    mask = mask & tf.transpose(mask, perm = (1, 0))
    #mask = tf.ones((5, 5))
    #mask = (tf.linalg.band_part(mask, -1, 0) == 1)
    print(mask)
    _scores = tf.where(mask == False, tf.fill(tf.shape(data), -1e9), data)
    print(data, mask, _scores)

#mask_sample()
#exit()    
  
src_vocab_size = 5000
tgt_vocab_size = 5000

batch_size = 32
d_model = 32
seq_length = 10
header = 4
stack_size = 2

'''
emb = EmbeddingLayer(vocab_size, d_model)
pelayer = PositionalEncoding(d_model)
x = keras.Input((seq_length,), batch_size)
#x = tf.ones((batch_size, seq_length), dtype=tf.int64)
x_out = emb(x)
print(x.shape, x.dtype)
print(x_out.shape, x_out.dtype)
y = pelayer(x_out)
'''

_data = English2FrenchData(batch_size, seq_length)
#val_data = English2FrenchData(batch_size, seq_length)
src_vocab_size = len(_data.src_vocab)
tgt_vocab_size = len(_data.tgt_vocab)

params = (src_vocab_size, tgt_vocab_size, d_model, header, stack_size)
model = Transformer(*params, src_pad_idx = 4, tgt_pad_idx = 5)
#model.build_graph(seq_length).summary()

#model.build((vocab_size, d_model))
#model.summary()
#_x_enc = tf.rand((batch_size, seq_length))
#_x_dec = tf.ones((batch_size, seq_length))

'''
_x_enc = tf.random.uniform((batch_size, seq_length), 0, src_vocab_size, dtype=tf.int32, seed=0)
_x_dec = tf.random.uniform((batch_size, seq_length), 0, src_vocab_size, dtype=tf.int32, seed=0)
_enc_mask = model.make_src_mask(_x_enc)
_dec_mask = model.make_trg_mask(_x_dec)

output = model(_x_enc, _x_dec)
print(output, _x_enc)
'''
#v = model.make_trg_mask(_x_enc)
#print(v, v.shape)

train_iter = _data.get_dataloader(train=True)
val_iter = _data.get_dataloader(train=False)

learning_rate = 0.005

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

BUFFER_SIZE = 20000
BATCH_SIZE = batch_size

MAX_TOKENS=128
def prepare_batch(enc, dec, label):
    return [enc, dec], label
  
def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)      
      .prefetch(buffer_size=tf.data.AUTOTUNE))
  
model.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

# Create training and validation set batches.
train_batches = make_batches(train_iter)
val_batches = make_batches(val_iter)

model.fit(train_batches,
                epochs=10,
                validation_data=val_batches)