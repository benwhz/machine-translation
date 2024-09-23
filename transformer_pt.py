import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from dataset_pt import English2FrenchData
import time

torch.set_printoptions(precision=5, sci_mode=False)
torch.manual_seed(0)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model) -> None:
        '''
        Args:
            vacab_size:     size of vocabulary
            d_model:        dimension of embeddings
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.emb_table = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        '''
        Args:
            x:      input tensor, shape = (batch_size, seq_length)
        
        Return: 
            embedding vectors, shape = (batch_size, seq_length, d_model)
        '''
        x = x.long()
        #print(10*'*', x.shape)

        return self.emb_table(x) * math.sqrt(self.d_model)
   
class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float, max_length: int = 5000):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    # inherit from Module
    super().__init__()     

    # initialize dropout                  
    self.dropout = nn.Dropout(p=dropout)      

    # create tensor of 0s
    pe = torch.zeros(max_length, d_model)    

    # create position column   
    k = torch.arange(0, max_length)
    k = k.unsqueeze(1)  

    # calc divisor for positional encoding 
    div_term = torch.exp(                                 
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )

    d = torch.sin(k * div_term)
    
    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)    

    # calc cosine on odd indices   
    pe[:, 1::2] = torch.cos(k * div_term)  

    # add dimension     
    pe = pe.unsqueeze(0)          

    # buffers are saved in state_dict but not trained by the optimizer                        
    self.register_buffer("pe", pe)                        

  def forward(self, x):
    """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)
    
    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    # add positional encoding to the embeddings
    # print(10*'+', x.shape)
    x = x + self.pe[:, : x.size(1)].requires_grad_(False) 
    # print(10*'+', x.shape)
    # perform dropout
    return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, model_size, headers, dropout) -> None:
        super().__init__()
        self.W_q = nn.Linear(model_size, model_size)
        self.W_k = nn.Linear(model_size, model_size)
        self.W_v = nn.Linear(model_size, model_size)
        self.headers = headers
        self.query_size = int(model_size/headers)
        self.W_o = nn.Linear(model_size, model_size)
        self.dropout = nn.Dropout(p=dropout)      
        
    def _seperate_by_header(self, v):
        batch_size, seq_length, model_size = v.shape
        return v.reshape(batch_size, seq_length, self.headers, self.query_size)\
                .permute(0, 2, 1, -1)
                #.reshape(batch_size*self.headers, seq_length, -1)

    def _aggregate_by_head(self, v):
        _batch, header, seq_length, _dim = v.shape
        #batch_size = int(_batch/self.headers)
        return v.permute(0, 2, 1, -1)\
                .reshape(_batch, seq_length, -1)
                
    def forward(self, query, keys, mask=None):
        # query, keys shape = (batch_size, seq_length, model_size) (3, 5, 32)
        batch_size = query.shape[0]
        seq_length = query.shape[1]
        # (batch_size, seq_length, model_size) => (batch_size*self.headers, seq_length, self.query_size)
        # (3, 5, 32) => (3, 4, 5, 8)
        
        _query = self._seperate_by_header(self.W_q(query))
        _keys = self._seperate_by_header(self.W_k(keys))
        values = self._seperate_by_header(self.W_v(keys))
        
        # (3, 4, 5, 8) @ (3, 4, 8, 5) = (3, 4, 5, 5)
        _scores = _query@_keys.permute(0, 1, 3, 2) / math.sqrt(self.query_size)
        
        # mask shape = (3, 1, 1, 5) or (3, 1, 5, 5)
        # print('attention mask shape = ', mask.shape, _scores.shape)
        if mask is not None:
            _scores = _scores.masked_fill(mask == 0, -1e9)

        # weights shape = (3, 3, 5, 5)
        _weights = self.dropout(F.softmax(_scores, dim = -1))
    
        # (3, 4, 5, 5) @ (3, 4, 5, 8) = (3, 4, 5, 8)
        context = _weights @ values

        # (3, 4, 5, 8) => (3, 5, 32)
        context = self._aggregate_by_head(context)
        context = self.W_o(context)

        #_weights = _weights.reshape(batch_size, )
        
        return context

class FeedForwardLayer(nn.Module):
  def __init__(self, d_model, dropout, d_ffn = 2048 ) -> None:
      super().__init__()
      self.liner1 = nn.Linear(d_model, d_ffn)
      self.relu = nn.ReLU()
      self.liner2 = nn.Linear(d_ffn, d_model)
      self.dropout = nn.Dropout(p=dropout)      

  def forward(self, x):
    return self.liner2(self.dropout(self.relu(self.liner1(x))))     

class EncodeBlock(nn.Module):
    def __init__(self, d_model, header, dropout) -> None:
      super().__init__()         
      self.attention = MultiHeadAttention(d_model, header, dropout)
      self.norm1 = nn.LayerNorm(d_model)
      self.ffn = FeedForwardLayer(d_model, dropout)
      self.norm2 = nn.LayerNorm(d_model)
      self.dropout = nn.Dropout(p=dropout)      

    def forward(self, x, mask):
       x = self.norm1(x + self.dropout(self.attention(x, x, mask)))
       x = self.norm2(x + self.dropout(self.ffn(x)))
       return x

class DecodeBlock(nn.Module):
    def __init__(self, d_model, header, dropout) -> None:
      super().__init__()         
      self.attention1 = MultiHeadAttention(d_model, header, dropout)
      self.norm1 = nn.LayerNorm(d_model)
      self.attention2 = MultiHeadAttention(d_model, header, dropout)
      self.norm2 = nn.LayerNorm(d_model)
      self.ffn = FeedForwardLayer(d_model, dropout)
      self.norm3 = nn.LayerNorm(d_model)
      self.dropout = nn.Dropout(p=dropout)      

    def forward(self, x, context, src_mask, tgt_mask):
       x = self.norm1(x + self.dropout(self.attention1(x, x, tgt_mask)))
       x = self.norm2(x + self.dropout(self.attention2(x, context, src_mask)))
       x = self.norm3(x + self.dropout(self.ffn(x)))
       return x

class Encoder(nn.Module):
   def __init__(self, d_model, header, n_block, dropout: float = 0.1) -> None:
      super().__init__()
      self.blocks = nn.ModuleList([EncodeBlock(d_model, header, dropout) for n in range(n_block)])
      self.dropout = nn.Dropout(dropout)

   def forward(self, x, mask = None):
      for encode in self.blocks:
         x = encode(x, mask)

      return self.dropout(x)

class Decoder(nn.Module):
   def __init__(self, d_model, header, n_block, dropout: float = 0.1) -> None:
      super().__init__()
      self.blocks = nn.ModuleList([DecodeBlock(d_model, header, dropout) for n in range(n_block)])
      self.dropout = nn.Dropout(dropout)

   def forward(self, x, context, src_mask = None, tgt_mask = None):
      for decode in self.blocks:
         x = decode(x, context, src_mask, tgt_mask)

      return self.dropout(x)

class EndLayer(nn.Module):
   def __init__(self, vocab_size, d_model) -> None:
      super().__init__()
      self.liner = nn.Linear(d_model, vocab_size)
      self.softmax = nn.Softmax(dim=-1)

   def forward(self, x):
       x = self.liner(x)
       return x
      
class Transformer(nn.Module):
  def __init__(self, *args, **argv) -> None:
      super().__init__()
      src_vocab_size, tgt_vcab_size, d_model, header, stack_size = args
      self.enc_embedding = EmbeddingLayer(src_vocab_size, d_model)
      self.dec_embedding = EmbeddingLayer(tgt_vcab_size, d_model)
      
      self.pos_encoding = PositionalEncoding(d_model, 0.1)
      self.encoder = Encoder(d_model, header, stack_size, 0.1)
      self.decoder = Decoder(d_model, header, stack_size, 0.1)
      self.endl = EndLayer(tgt_vcab_size, d_model)

      self.src_pad_idx = 0
      if 'src_pad_idx' in argv.keys():
        self.src_pad_idx = argv['src_pad_idx']

      self.trg_pad_idx = 0
      if 'tgt_pad_idx' in argv.keys():
        self.trg_pad_idx = argv['tgt_pad_idx']
      
  def forward(self, *inputs):
      enc_inputs, dec_inputs = inputs
      _enc_mask = self.make_src_mask(enc_inputs)
      _dec_mask = self.make_trg_mask(dec_inputs)
      enc_x = self.enc_embedding(enc_inputs)
      enc_x = self.pos_encoding(enc_x)
      dec_x = self.dec_embedding(dec_inputs)
      dec_x = self.pos_encoding(dec_x)
      c = self.encoder(enc_x, _enc_mask)
      x = self.decoder(dec_x, c, _enc_mask, _dec_mask)
      return self.endl(x)

  def make_src_mask(self, src: torch.Tensor):
    """
    Args:
        src:          raw sequences with padding        (batch_size, seq_length)              
    
    Returns:
        src_mask:     mask for each sequence            (batch_size, 1, 1, seq_length)
    """
    # assign 1 to tokens that need attended to and 0 to padding tokens, then add 2 dimensions
    # src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    src_mask = (src != self.src_pad_idx).unsqueeze(1)
    src_mask = src_mask & src_mask.permute(0, 2, 1)
    src_mask = src_mask.unsqueeze(1)
    return src_mask

  def make_trg_mask(self, trg: torch.Tensor):
    """
    Args:
        trg:          raw sequences with padding        (batch_size, seq_length)              
    
    Returns:
        trg_mask:     mask for each sequence            (batch_size, 1, seq_length, seq_length)
    """

    seq_length = trg.shape[1]

    # assign True to tokens that need attended to and False to padding tokens, then add 2 dimensions
    trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_length)

    # generate subsequent mask
    trg_sub_mask = torch.tril(torch.ones((seq_length, seq_length))).bool() # (batch_size, 1, seq_length, seq_length)

    # bitwise "and" operator | 0 & 0 = 0, 1 & 1 = 1, 1 & 0 = 0
    trg_mask = trg_mask & trg_sub_mask

    return trg_mask
    
def mask_sample():
    data = torch.ones((3, 4, 5, 5))
    mask = torch.tensor([[True, True, True, True, False]])
    mask = mask & mask.permute(1, 0)
    #mask[-1, :] = False
    #mask = torch.tril(torch.ones((5, 5))).bool()
    print(mask, mask.shape)
    _scores = data.masked_fill(mask == 0, -1e9)        
    print(data, mask, _scores)

#mask_sample()
#exit()
#d = torch.tensor([-1e10,-1e10,-1e10,-1e10])  
#s=F.softmax(d)  
#print(s)

src_vocab_size = 5000
tgt_vocab_size = 5000

batch_size = 32
d_model = 32
seq_length = 10
header = 4
stack_size = 2

'''
emb = EmbeddingLayer(vocab_size, d_model)
x = torch.ones((batch_size, seq_length), dtype=torch.long)
x_out = emb(x)
print(x.shape, x.dtype)
print(x_out.shape)
pelayer = PositionalEncoding(d_model)
print(pelayer(x_out)[0, 0])
'''
train_data = English2FrenchData(batch_size, seq_length)
val_data = English2FrenchData(batch_size, seq_length)
src_vocab_size = len(train_data.src_vocab)
tgt_vocab_size = len(train_data.tgt_vocab)

params = (src_vocab_size, tgt_vocab_size, d_model, header, stack_size)
model = Transformer(*params, src_pad_idx = train_data.src_vocab.token_to_idx['<pad>'], tgt_pad_idx = train_data.tgt_vocab.token_to_idx['<pad>'])
# print(model)
# summary(model, [(seq_length,), (seq_length,)], batch_size)

'''
_x_enc = torch.randint(src_vocab_size, (batch_size, seq_length))
_x_dec = torch.randint(tgt_vocab_size, (batch_size, seq_length))
_x_enc[0, -3:] = 0
#_enc_mask = model.make_src_mask(_x_enc)
#_dec_mask = model.make_trg_mask(_x_dec)

#input = (_x_enc, _x_dec)
#output = model(*input)
output = model(_x_enc, _x_dec)
print(output, _x_enc)

#v = model.make_src_mask(_x_enc)
#print(v, v.shape)
'''

PAD_IDX = train_data.src_vocab.token_to_idx['<pad>']
LEARNING_RATE = 0.005

train_iter = train_data.get_dataloader(train=True)
val_iter = val_data.get_dataloader(train=False)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
  """
    Train the model on the given data.

    Args:
        model:        Transformer model to be trained
        iterator:     data to be trained on
        optimizer:    optimizer for updating parameters
        criterion:    loss function for updating parameters
        clip:         value to help prevent exploding gradients

    Returns:
        loss for the epoch
  """

  # set the model to training mode
  model.train()
    
  epoch_loss = 0
    
  # loop through each batch in the iterator
  for i, batch in enumerate(iterator):

    # set the source and target batches    
    src, trg, _, real = batch
        
    # zero the gradients
    optimizer.zero_grad()
        
    # logits for each output
    logits = model(src, trg)

    # expected output
    expected_output = real
    
    #print(logits.shape, expected_output.shape)
    #print(logits.contiguous().view(-1, logits.shape[-1]).shape, expected_output.contiguous().view(-1).shape)
    
    # logtis shape = (batch_size, seq_length, vocab_size)
    # real shape = (batch_size, seq_length)
    # ==> loss calculation shape (batch_size*seq_length, vacab_size) and (batch_size*seq_length)
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # calculate the loss
    loss = criterion(logits.contiguous().view(-1, logits.shape[-1]), 
                    expected_output.contiguous().view(-1))
      
    # backpropagation
    loss.backward()
        
    # clip the weights
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
    # update the weights
    optimizer.step()
        
    # update the loss
    epoch_loss += loss.item()

  # return the average loss for the epoch
  return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
  """
    Evaluate the model on the given data.

    Args:
        model:        Transformer model to be trained
        iterator:     data to be evaluated
        criterion:    loss function for assessing outputs

    Returns:
        loss for the data
  """

  # set the model to evaluation mode
  model.eval()
    
  epoch_loss = 0
    
  # evaluate without updating gradients
  with torch.no_grad():
    
    # loop through each batch in the iterator
    for i, batch in enumerate(iterator):
      
      # set the source and target batches  
      src, trg, _, real = batch


      # logits for each output
      logits = model(src, trg)

      # expected output
      expected_output = real
    
      # calculate the loss
      loss = criterion(logits.contiguous().view(-1, logits.shape[-1]), 
                      expected_output.contiguous().view(-1))

      # update the loss
      epoch_loss += loss.item()
        
  # return the average loss for the epoch
  return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

# loop through each epoch
for epoch in range(N_EPOCHS):
    
  start_time = time.time()
    
  # calculate the train loss and update the parameters
  train_loss = train(model, train_iter, optimizer, criterion, CLIP)

  # calculate the loss on the validation set
  valid_loss = evaluate(model, val_iter, criterion)
    
  end_time = time.time()
    
  # calculate how long the epoch took
  epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
  # save the model when it performs better than the previous run
  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), '../output/transformer-model.pt')
    
  print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')