import torch
import math
from torch import nn

DROPOUT = 0.1
MAX_LEN = 5000

def scaled_dot_product_attention(Q: torch.Tensor,
                                 K: torch.Tensor, 
                                 V: torch.Tensor,
                                 mask=None):
    """    
    Args:
        Q: Query  (batch, heads, seq_q, d_k)
        K: Key    (batch, heads, seq_k, d_k)
        V: Value  (batch, heads, seq_k, d_v)
        mask: Optional, can be broadcast to (batch, heads, seq_q, seq_k). Positions with mask == 0 will be masked out.
    Returns:
        output: (batch, heads, seq_q, d_v)
        attn_weights: (batch, heads, seq_q, seq_k)
    """
    d_k = Q.shape[-1]
    
    # scores: (batch, heads, seq_q, seq_k)
    # deviding scores by $\sqrt{d_k}$ to avoid a large number, resulting in saturated Softmax, and eventually leading to vanishing gradient.
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # mask: Apply a mask to the attention scores before the softmax operation to manually suppress specific positions, ensuring the model ignores irrelevant or future information.
    # to suppress specific positions, a musk manually set its score to Minus Infinity.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)    
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    Args:
        d_model: The dimensionality of the input and output feature vectors.
        num_heads: The number of parallel attention heads.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def split_heads(self, 
                    X: torch.Tensor):
        """
        split (batch, seq, d_model) into (batch, heads, seq, d_K)
        in which d_k = d_model // heads
        """
        batch, seq, _ = X.shape
        X = X.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        return X
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query (batch, seq_q, d_model)
            K: Key (batch, seq_k, d_model)
            V: Value (batch, seq_k, d_model)
            mask: can be broadcast to (batch, heads, seq_q, seq_k)
        """
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))
        
        x, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # concatenate (batch, num_heads, seq, d_k) into (batch, seq, d_model)
        batch, _, seq, _ = x.shape
        x = x.transpose(1, 2).reshape(batch, seq, self.d_model)
        
        return self.W_O(x), attn_weights
    
class FeedForward(nn.Module):
    """
    Args:
        d_model: The dimensionality of the input and output feature vectors.
        d_ff: The dimensionality of the hidden layer in the Feed Forward network.
    """
    def __init__(self, d_model, d_ff, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.net(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # Positional Encoding does not participate in Back Propagation
        
        """
        Why not a value with `require_grad` set to FALSE?
        PE is a state calculated solely by mathematic formula, rather than a nn.Parameter() not to be trained (E.g., GolVe).
        """
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=DROPOUT):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask=None):
        attn_out, _  = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=DROPOUT):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        attn_out, _ = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, num_layers, vocab_size, d_model, d_ff, num_heads, embedding = None, max_len=MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])
        
        """
        Note on Embedding Scaling:
        In the Transformer architecture, we multiply the learned embedding weights by $\sqrt{d_{\text{model}}}$ before adding the Positional Encoding. Since positional encodings typically range between $[-1, 1]$, scaling the embeddings ensures their magnitude is sufficiently large relative to the positional signals. This prevents the semantic information from being "washed out" by the positional data, allowing the model to effectively integrate both meaning and sequence order.
        """
        self.scale = math.sqrt(d_model)
        
    def forward(self, src, src_mask=None):
        x = self.embedding(src) * self.scale
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self, num_layers, vocab_size, d_model, d_ff, num_heads, embedding = None, max_len=MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])
        self.scale = math.sqrt(d_model)
    
    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt) * self.scale
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, src_vocab_size, tgt_vocab_size, d_model, d_ff, num_heads, max_len=MAX_LEN, dropout=DROPOUT, bi_embedded=False, full_embedded=False):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = self.src_embedding if full_embedded is True else nn.Embedding(tgt_vocab_size, d_model)
        
        self.encoder = Encoder(num_layers, src_vocab_size, d_model, d_ff, num_heads, self.src_embedding, max_len, dropout)
        self.decoder = Decoder(num_layers, tgt_vocab_size, d_model, d_ff, num_heads, self.tgt_embedding, max_len, dropout)
        
        # we set bias to False, in order to match the weight of embedding layer        
        self._init_weights()
        
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        if bi_embedded is True or full_embedded is True:
            self.output_proj.weight = self.tgt_embedding.weight
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim()> 1:
                nn.init.xavier_uniform_(p)
                
    def make_src_mask(self,
                      src: torch.Tensor,
                      pad_idx=0):
        """
        mask out padding token <PAD>
        Args:
            src: (batch, seq)
        returns:
            (batch, 1, 1, seq)
            can be broadcast to (batch, heads, seq_q, seq_k)
        """
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self,
                      tgt: torch.Tensor,
                      pad_idx=0):
        """
        mask out padding token <PAD> and future token
        Args:
            tgt: (batch, seq)
        returns:
            (batch, 1, seq, seq)
            which is a lower-traingle matrix
        """
        seq = tgt.shape[-1]
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(seq, seq, device=tgt.device)).bool()
        return pad_mask & causal_mask
    
    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        src_mask = self.make_src_mask(src, src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, tgt_pad_idx)
        
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        
        return self.output_proj(dec_out) # (batch, seq, vocab_size)

class TransformerClassifier(nn.Module):
    def __init__(self, num_layers, vocab_size, d_model, d_ff, num_heads, num_classes=2, max_len=MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.encoder = Encoder(num_layers, vocab_size, d_model, d_ff, num_heads, None, max_len, dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self.make_mask = lambda src, pad_idx=0: (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, src, pad_idx=0):
        src_mask = self.make_mask(src, pad_idx)
        enc_out = self.encoder(src, src_mask)
        
        pooled_out = torch.mean(enc_out, dim=1) 
        
        return self.classifier(pooled_out)