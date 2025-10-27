
# Created at
# 2025-10-11

# Change Log
# 2025-10-11: Docstring, TypeHint, Pre-normalization
# 2025-10-11: Speed optimization (cp.softmax), attention numerics clarification, mask handling clarification
# 2025-10-11: MultiHeadAttention batch-wise matmul clarification, attention mask shape/dtype clarification, LayerNorm float64, Dropout mask regeneration clarification
# 2025-10-11: Learnable affine LayerNorm, Embedding update (via parameters() method), SGD optimizer integration
# 2025-10-11: Fixed missing enc_layers and dec_layers initialization in Encoder and Decoder __init__
# 2025-10-11: Replaced cp.softmax with cupy.nn.softmax to resolve AttributeError
# 2025-10-11: Reverted softmax to manual implementation due to ModuleNotFoundError for cupy.nn

import cupy as cp
import cupy.random as random

class Parameter:
  def __init__(self, data: cp.ndarray):
    self.data = data
    self.grad = None

  def __repr__(self):
    return f"Parameter(shape={self.data.shape}, dtype={self.data.dtype})"

  # Allow direct operations on the data for convenience
  def __array__(self):
    return self.data

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    # Delegate ufunc operations to the underlying data
    processed_inputs = [i.data if isinstance(i, Parameter) else i for i in inputs]
    return getattr(self.data, '__array_ufunc__')(ufunc, method, *processed_inputs, **kwargs)

  def __getattr__(self, name):
    # Delegate attribute access to the underlying data if not found in Parameter
    return getattr(self.data, name)

  def __setattr__(self, name, value):
    # Allow setting 'data' and 'grad' directly, otherwise delegate to data
    if name in ('data', 'grad'):
      object.__setattr__(self, name, value)
    else:
      setattr(self.data, name, value)


class SGD:
  def __init__(self, params: list[Parameter], lr: float = 0.01):
    self.params = params
    self.lr = lr

  def step(self):
    for param in self.params:
      if hasattr(param, 'grad') and param.grad is not None:
        param -= self.lr * param.grad

class Dropout:
  def __init__(self, rate: float):
    self.rate = rate

  def forward(self, x: cp.ndarray, training: bool) -> cp.ndarray:
    if not training or self.rate == 0:
      return x
    
    # Generate a random mask for each training step.
    # This ensures different neurons are dropped out in each forward pass,
    # which is crucial for preventing co-adaptation of neurons.
    mask = random.rand(*x.shape) > self.rate
    
    # Scale the output by 1 / (1 - rate)
    return (x * mask) / (1 - self.rate)

class LayerNormalization:
  def __init__(self, d_model: int, epsilon: float = 1e-6):
    self.epsilon = cp.array(epsilon, dtype=cp.float64)
    # Learnable affine parameters
    self.gamma = Parameter(cp.ones(d_model, dtype=cp.float64))
    self.beta = Parameter(cp.zeros(d_model, dtype=cp.float64))

  def forward(self, x: cp.ndarray) -> cp.ndarray:
    # Ensure calculations are done in float64 for higher precision
    x_f64 = x.astype(cp.float64)
    mean = cp.mean(x_f64, axis=-1, keepdims=True)
    variance = cp.var(x_f64, axis=-1, keepdims=True)
    # Normalize
    normalized_x = (x_f64 - mean) / cp.sqrt(variance + self.epsilon)
    # Apply learnable affine transformation
    return (normalized_x * self.gamma + self.beta).astype(x.dtype)

  def parameters(self) -> list[Parameter]:
    return [self.gamma, self.beta]

class PositionalEncoding:
  """
  Implements positional encoding for Transformer models.

  Positional encoding adds information about the position of each token
  in the sequence, which is crucial for Transformer models that do not
  inherently process sequence order.

  Args:
    d_model (int): The dimensionality of the model's output.
    max_seq_len (int): The maximum sequence length the model will handle.
  """
  def __init__(self, d_model: int, max_seq_len: int):
    self.d_model = d_model
    self.max_seq_len = max_seq_len
    self.positional_encoding_matrix = self._get_positional_encoding_matrix()

  def _get_positional_encoding_matrix(self) -> cp.ndarray:
    """
    Generates the positional encoding matrix.

    Returns:
      cp.ndarray: A matrix of shape (1, max_seq_len, d_model)
                  containing the positional encodings.
    """
    position = cp.arange(self.max_seq_len)[:, cp.newaxis]
    div_term = cp.exp(
      cp.arange(0, self.d_model, 2) * -(cp.log(10000.0) / self.d_model)
    )
    pe = cp.zeros((self.max_seq_len, self.d_model))
    pe[:, 0::2] = cp.sin(position * div_term)
    pe[:, 1::2] = cp.cos(position * div_term)
    return pe[cp.newaxis, :, :].copy()

  def forward(self, x: cp.ndarray) -> cp.ndarray:
    """
    Adds positional encoding to the input tensor.

    Args:
      x (cp.ndarray): Input tensor with shape (batch_size, seq_len, d_model).

    Returns:
      cp.ndarray: Input tensor with positional encoding added.
    """
    # x: (batch_size, seq_len, d_model)
    seq_len = x.shape[1]
    return x + self.positional_encoding_matrix[:, :seq_len, :]


class MultiHeadAttention:
  """
  Implements Multi-Head Attention mechanism.

  Args:
    d_model (int): The dimensionality of the model's output.
    num_heads (int): The number of attention heads.
    dropout_rate (float): The dropout rate to apply.
  """
  def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads

    self.wq = Parameter(cp.random.randn(d_model, d_model) * cp.sqrt(2.0 / d_model))
    self.wk = Parameter(cp.random.randn(d_model, d_model) * cp.sqrt(2.0 / d_model))
    self.wv = Parameter(cp.random.randn(d_model, d_model) * cp.sqrt(2.0 / d_model))
    self.wo = Parameter(cp.random.randn(d_model, d_model) * cp.sqrt(2.0 / d_model))

    self.dropout = Dropout(dropout_rate)

  def parameters(self) -> list[Parameter]:
    return [self.wq, self.wk, self.wv, self.wo]

  def _scaled_dot_product_attention(
    self, q: cp.ndarray, k: cp.ndarray, v: cp.ndarray, mask: cp.ndarray = None
  ) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Calculates scaled dot-product attention.

    Args:
      q (cp.ndarray): Query tensor with shape (batch_size, num_heads, seq_len_q, head_dim).
      k (cp.ndarray): Key tensor with shape (batch_size, num_heads, seq_len_k, head_dim).
      v (cp.ndarray): Value tensor with shape (batch_size, num_heads, seq_len_v, head_dim).
      mask (cp.ndarray, optional): An additive attention mask. Shape can be (batch_size, 1, 1, seq_len_k) for padding or (batch_size, 1, seq_len_q, seq_len_k) for look-ahead. Dtype is typically cp.float64, where masked positions are -inf (or a very large negative number like -1e9) and unmasked positions are 0. Defaults to None.

    Returns:
      tuple[cp.ndarray, cp.ndarray]: A tuple containing:
        - cp.ndarray: Output of the attention mechanism.
        - cp.ndarray: Attention weights.
    """
    # q, k, v: (batch_size, num_heads, seq_len, head_dim)
    matmul_qk = cp.matmul(
      q, k.swapaxes(-2, -1)
    )  # (batch_size, num_heads, seq_len, seq_len)
    dk = cp.sqrt(cp.array(k.shape[-1], dtype=cp.float32))
    scaled_attention_logits = matmul_qk / dk

    if mask is not None:
      # Add a large negative number to masked positions. This makes their softmax probability close to 0.
      scaled_attention_logits += (mask * -1e9)

    # Apply softmax to get attention weights. The subtraction of max_logits is for numerical stability.
    exp_logits = cp.exp(scaled_attention_logits - cp.max(scaled_attention_logits, axis=-1, keepdims=True))
    attention_weights = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)
    output = cp.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
    return output, attention_weights

  def forward(
    self, 
    q: cp.ndarray, 
    k: cp.ndarray, 
    v: cp.ndarray, 
    mask: cp.ndarray = None, 
    training: bool = True
  ) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Forward pass for Multi-Head Attention.

    Args:
      q (cp.ndarray): Query tensor with shape (batch_size, seq_len_q, d_model).
      k (cp.ndarray): Key tensor with shape (batch_size, seq_len_k, d_model).
      v (cp.ndarray): Value tensor with shape (batch_size, seq_len_v, d_model).
      mask (cp.ndarray, optional): An additive attention mask. Shape can be (batch_size, 1, 1, seq_len_k) for padding or (batch_size, 1, seq_len_q, seq_len_k) for look-ahead. Dtype is typically cp.float64, where masked positions are -inf (or a very large negative number like -1e9) and unmasked positions are 0. Defaults to None.
      training (bool): Whether the model is in training mode.

    Returns:
      tuple[cp.ndarray, cp.ndarray]: A tuple containing:
        - cp.ndarray: Output tensor with shape (batch_size, seq_len_q, d_model).
        - cp.ndarray: Attention weights.
    """
    batch_size = q.shape[0]

    # Linear layers and split into heads. These operations are already vectorized
    # and performed efficiently across the batch and sequence dimensions.
    q = (
      cp.matmul(q, self.wq)
      .reshape(batch_size, -1, self.num_heads, self.head_dim)
      .swapaxes(1, 2)
    )
    k = (
      cp.matmul(k, self.wk)
      .reshape(batch_size, -1, self.num_heads, self.head_dim)
      .swapaxes(1, 2)
    )
    v = (
      cp.matmul(v, self.wv)
      .reshape(batch_size, -1, self.num_heads, self.head_dim)
      .swapaxes(1, 2)
    )

    # Scaled dot-product attention
    scaled_attention, attention_weights = self._scaled_dot_product_attention(
      q, k, v, mask
    )
    scaled_attention = self.dropout.forward(scaled_attention, training)

    # Concatenate heads and put through final linear layer
    scaled_attention = scaled_attention.swapaxes(1, 2).reshape(
      batch_size, -1, self.d_model
    )
    output = cp.matmul(scaled_attention, self.wo)
    return output, attention_weights


class FeedForwardNetwork:
  """
  Implements the position-wise feed-forward network.

  Args:
    d_model (int): The dimensionality of the model's output.
    d_ff (int): The dimensionality of the inner layer of the feed-forward network.
    dropout_rate (float): The dropout rate to apply.
  """
  def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
    self.w1 = Parameter(cp.random.randn(d_model, d_ff) * cp.sqrt(2.0 / d_model))
    self.b1 = Parameter(cp.zeros(d_ff))
    self.w2 = Parameter(cp.random.randn(d_ff, d_model) * cp.sqrt(2.0 / d_ff))
    self.b2 = Parameter(cp.zeros(d_model))
    self.dropout = Dropout(dropout_rate)

  def parameters(self) -> list[Parameter]:
    return [self.w1, self.b1, self.w2, self.b2]

  def forward(self, x: cp.ndarray, training: bool = True) -> cp.ndarray:
    """
    Forward pass for the feed-forward network.

    Args:
      x (cp.ndarray): Input tensor with shape (batch_size, seq_len, d_model).
      training (bool): Whether the model is in training mode.

    Returns:
      cp.ndarray: Output tensor with shape (batch_size, seq_len, d_model).
    """
    # x: (batch_size, seq_len, d_model)
    x = cp.maximum(0, cp.matmul(x, self.w1) + self.b1).astype(cp.float32)  # ReLU activation
    x = self.dropout.forward(x, training)
    x = cp.matmul(x, self.w2) + self.b2
    x = self.dropout.forward(x, training)
    return x


class EncoderLayer:
  """
  Represents a single encoder layer in the Transformer architecture.

  Args:
    d_model (int): The dimensionality of the model's output.
    num_heads (int): The number of attention heads.
    d_ff (int): The dimensionality of the feed-forward network's inner layer.
    dropout_rate (float): The dropout rate to apply.
  """
  def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
    self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
    self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)

    self.layernorm1 = LayerNormalization(d_model)
    self.layernorm2 = LayerNormalization(d_model)

    self.dropout1 = Dropout(dropout_rate)
    self.dropout2 = Dropout(dropout_rate)

  def parameters(self) -> list[cp.ndarray]:
    params = []
    params.extend(self.mha.parameters())
    params.extend(self.ffn.parameters())
    params.extend(self.layernorm1.parameters())
    params.extend(self.layernorm2.parameters())
    return params

  def forward(self, x: cp.ndarray, mask: cp.ndarray, training: bool = True) -> cp.ndarray:
    """
    Forward pass for the encoder layer.

    Args:
      x (cp.ndarray): Input tensor with shape (batch_size, seq_len, d_model).
      mask (cp.ndarray): An additive padding mask. Shape is typically (batch_size, 1, 1, seq_len). Dtype is typically cp.float64, where masked positions are -inf (or a very large negative number like -1e9) and unmasked positions are 0.
      training (bool): Whether the model is in training mode.

    Returns:
      cp.ndarray: Output tensor with shape (batch_size, seq_len, d_model).
    """
    # Pre-normalization for self-attention
    norm_x = self.layernorm1.forward(x)
    attn_output, _ = self.mha.forward(norm_x, norm_x, norm_x, mask, training)
    attn_output = self.dropout1.forward(attn_output, training)
    out1 = x + attn_output  # Residual connection

    # Pre-normalization for feed-forward network
    norm_out1 = self.layernorm2.forward(out1)
    ffn_output = self.ffn.forward(norm_out1, training)
    ffn_output = self.dropout2.forward(ffn_output, training)
    out2 = out1 + ffn_output  # Residual connection
    return out2


class DecoderLayer:
  """
  Represents a single decoder layer in the Transformer architecture.

  Args:
    d_model (int): The dimensionality of the model's output.
    num_heads (int): The number of attention heads.
    d_ff (int): The dimensionality of the feed-forward network's inner layer.
    dropout_rate (float): The dropout rate to apply.
  """
  def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
    self.mha1 = MultiHeadAttention(
      d_model, num_heads, dropout_rate
    )  # Masked self-attention
    self.mha2 = MultiHeadAttention(
      d_model, num_heads, dropout_rate
    )  # Encoder-decoder attention
    self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)

    self.layernorm1 = LayerNormalization(d_model)
    self.layernorm2 = LayerNormalization(d_model)
    self.layernorm3 = LayerNormalization(d_model)

    self.dropout1 = Dropout(dropout_rate)
    self.dropout2 = Dropout(dropout_rate)
    self.dropout3 = Dropout(dropout_rate)

  def parameters(self) -> list[cp.ndarray]:
    params = []
    params.extend(self.mha1.parameters())
    params.extend(self.mha2.parameters())
    params.extend(self.ffn.parameters())
    params.extend(self.layernorm1.parameters())
    params.extend(self.layernorm2.parameters())
    params.extend(self.layernorm3.parameters())
    return params

  def forward(
    self, 
    x: cp.ndarray, 
    enc_output: cp.ndarray, 
    look_ahead_mask: cp.ndarray, 
    padding_mask: cp.ndarray, 
    training: bool = True
  ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Forward pass for the decoder layer.

    Args:
      x (cp.ndarray): Input tensor with shape (batch_size, seq_len, d_model).
      enc_output (cp.ndarray): Output from the encoder with shape (batch_size, inp_seq_len, d_model).
      look_ahead_mask (cp.ndarray): An additive look-ahead mask. Shape is typically (batch_size, 1, tar_seq_len, tar_seq_len). Dtype is typically cp.float64, where masked positions are -inf (or a very large negative number like -1e9) and unmasked positions are 0.
      padding_mask (cp.ndarray): An additive padding mask. Shape is typically (batch_size, 1, 1, inp_seq_len). Dtype is typically cp.float64, where masked positions are -inf (or a very large negative number like -1e9) and unmasked positions are 0.
      training (bool): Whether the model is in training mode.

    Returns:
      tuple[cp.ndarray, cp.ndarray, cp.ndarray]: A tuple containing:
        - cp.ndarray: Output tensor with shape (batch_size, seq_len, d_model).
        - cp.ndarray: Attention weights from the masked self-attention block.
        - cp.ndarray: Attention weights from the encoder-decoder attention block.
    """
    # Pre-normalization for masked self-attention
    norm_x = self.layernorm1.forward(x)
    attn1, attn_weights_block1 = self.mha1.forward(norm_x, norm_x, norm_x, look_ahead_mask, training)
    attn1 = self.dropout1.forward(attn1, training)
    out1 = x + attn1

    # Pre-normalization for encoder-decoder attention
    norm_out1 = self.layernorm2.forward(out1)
    attn2, attn_weights_block2 = self.mha2.forward(
      norm_out1, enc_output, enc_output, padding_mask, training
    )
    attn2 = self.dropout2.forward(attn2, training)
    out2 = out1 + attn2

    # Pre-normalization for feed-forward network
    norm_out2 = self.layernorm3.forward(out2)
    ffn_output = self.ffn.forward(norm_out2, training)
    ffn_output = self.dropout3.forward(ffn_output, training)
    out3 = out2 + ffn_output
    return out3, attn_weights_block1, attn_weights_block2


class Encoder:
  """
  The Transformer Encoder, composed of multiple EncoderLayers.

  Args:
    num_layers (int): The number of encoder layers.
    d_model (int): The dimensionality of the model's output.
    num_heads (int): The number of attention heads.
    d_ff (int): The dimensionality of the feed-forward network's inner layer.
    vocab_size (int): The size of the input vocabulary.
    max_seq_len (int): The maximum sequence length the model will handle.
    dropout_rate (float): The dropout rate to apply.
  """
  def __init__(
    self,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    vocab_size: int,
    max_seq_len: int,
    dropout_rate: float = 0.1,
  ):
    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = Parameter(cp.random.randn(vocab_size, d_model) * cp.sqrt(
      2.0 / vocab_size
    ))
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

    self.enc_layers = [
      EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
      for _ in range(num_layers)
    ]

    self.dropout = Dropout(dropout_rate)

  def parameters(self) -> list[Parameter]:
    params = [self.embedding]
    for layer in self.enc_layers:
      params.extend(layer.parameters())
    return params

  def forward(self, x: cp.ndarray, mask: cp.ndarray, training: bool = True) -> cp.ndarray:
    """
    Forward pass for the Encoder.

    Args:
      x (cp.ndarray): Input tensor with shape (batch_size, seq_len).
      mask (cp.ndarray): An additive padding mask for the encoder input. Shape is typically (batch_size, 1, 1, inp_seq_len). Dtype is typically cp.float64, where masked positions are -inf (or a very large negative number like -1e9) and unmasked positions are 0.
      training (bool): Whether the model is in training mode.

    Returns:
      cp.ndarray: Output tensor from the encoder with shape (batch_size, seq_len, d_model).
    """
    seq_len = x.shape[1]
    x = cp.take(self.embedding, x, axis=0)  # (batch_size, seq_len, d_model)
    x *= cp.sqrt(cp.array(self.d_model, dtype=cp.float32))
    x = self.pos_encoding.forward(x)
    x = self.dropout.forward(x, training)

    for i in range(self.num_layers):
      x = self.enc_layers[i].forward(x, mask, training)
    return x


class Decoder:
  """
  The Transformer Decoder, composed of multiple DecoderLayers.

  Args:
    num_layers (int): The number of decoder layers.
    d_model (int): The dimensionality of the model's output.
    num_heads (int): The number of attention heads.
    d_ff (int): The dimensionality of the feed-forward network's inner layer.
    vocab_size (int): The size of the target vocabulary.
    max_seq_len (int): The maximum sequence length the model will handle.
    dropout_rate (float): The dropout rate to apply.
  """
  def __init__(
    self,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    vocab_size: int,
    max_seq_len: int,
    dropout_rate: float = 0.1,
  ):
    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = Parameter(cp.random.randn(vocab_size, d_model) * cp.sqrt(
      2.0 / vocab_size
    ))
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

    self.dec_layers = [
      DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
      for _ in range(num_layers)
    ]

    self.dropout = Dropout(dropout_rate)

  def parameters(self) -> list[Parameter]:
    params = [self.embedding]
    for layer in self.dec_layers:
      params.extend(layer.parameters())
    return params

  def forward(
    self,
    x: cp.ndarray,
    enc_output: cp.ndarray,
    look_ahead_mask: cp.ndarray,
    padding_mask: cp.ndarray,
    training: bool = True,
  ) -> tuple[cp.ndarray, dict[str, cp.ndarray]]:
    """
    Forward pass for the Decoder.

    Args:
      x (cp.ndarray): Target input tensor with shape (batch_size, tar_seq_len).
      enc_output (cp.ndarray): Output from the encoder with shape (batch_size, inp_seq_len, d_model).
      look_ahead_mask (cp.ndarray): An additive look-ahead mask for the decoder self-attention. Shape is typically (batch_size, 1, tar_seq_len, tar_seq_len). Dtype is typically cp.float64, where masked positions are -inf (or a very large negative number like -1e9) and unmasked positions are 0.
      padding_mask (cp.ndarray): An additive padding mask for the encoder-decoder attention. Shape is typically (batch_size, 1, 1, inp_seq_len). Dtype is typically cp.float64, where masked positions are -inf (or a very large negative number like -1e9) and unmasked positions are 0.
      training (bool): Whether the model is in training mode.

    Returns:
      tuple[cp.ndarray, dict[str, cp.ndarray]]: A tuple containing:
        - cp.ndarray: Output tensor from the decoder with shape (batch_size, tar_seq_len, d_model).
        - dict: Dictionary of attention weights from each decoder layer.
    """
    seq_len = x.shape[1]
    x = cp.take(self.embedding, x, axis=0)  # (batch_size, seq_len, d_model)
    x *= cp.sqrt(cp.array(self.d_model, dtype=cp.float32))
    x = self.pos_encoding.forward(x)
    x = self.dropout.forward(x, training)

    attention_weights = {}
    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i].forward(
        x, enc_output, look_ahead_mask, padding_mask, training
      )
      attention_weights[f"decoder_layer{i+1}_block1"] = block1
      attention_weights[f"decoder_layer{i+1}_block2"] = block2
    return x, attention_weights


class TransformerArchitecture:
  """
  The complete Transformer model.

  Args:
    vocab_size (int): The size of the input/target vocabulary.
    d_model (int): The dimensionality of the model's output.
    num_heads (int): The number of attention heads.
    num_layers (int): The number of encoder and decoder layers.
    d_ff (int): The dimensionality of the feed-forward network's inner layer.
    max_seq_len (int): The maximum sequence length the model will handle.
    dropout_rate (float): The dropout rate to apply.
  """
  def __init__(
    self,
    vocab_size: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int,
    max_seq_len: int,
    dropout_rate: float = 0.1,
  ):
    self.encoder = Encoder(
      num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout_rate
    )
    self.decoder = Decoder(
      num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout_rate
    )
    self.final_layer = Parameter(cp.random.randn(d_model, vocab_size) * cp.sqrt(2.0 / d_model))

  def parameters(self) -> list[Parameter]:
    params = []
    params.extend(self.encoder.parameters())
    params.extend(self.decoder.parameters())
    params.append(self.final_layer)
    return params

  def forward(
    self,
    inp: cp.ndarray,
    tar: cp.ndarray,
    enc_padding_mask: cp.ndarray,
    look_ahead_mask: cp.ndarray,
    dec_padding_mask: cp.ndarray,
    training: bool = True,
  ) -> tuple[cp.ndarray, dict[str, cp.ndarray]]:
    """
    Forward pass for the Transformer model.

    Args:
      inp (cp.ndarray): Input sequence to the encoder with shape (batch_size, inp_seq_len).
      tar (cp.ndarray): Target sequence to the decoder with shape (batch_size, tar_seq_len).
      enc_padding_mask (cp.ndarray): An additive padding mask for the encoder input. Shape is typically (batch_size, 1, 1, inp_seq_len). Dtype is typically cp.float64, where masked positions are -inf (or a very large negative number like -1e9) and unmasked positions are 0.
      look_ahead_mask (cp.ndarray): An additive look-ahead mask for the decoder self-attention. Shape is typically (batch_size, 1, tar_seq_len, tar_seq_len). Dtype is typically cp.float64, where masked positions are -inf (or a very large negative number like -1e9) and unmasked positions are 0.
      dec_padding_mask (cp.ndarray): An additive padding mask for the decoder's encoder-decoder attention. Shape is typically (batch_size, 1, 1, inp_seq_len). Dtype is typically cp.float64, where masked positions are -inf (or a very large negative number like -1e9) and unmasked positions are 0.
      training (bool): Whether the model is in training mode.

    Returns:
      tuple[cp.ndarray, dict[str, cp.ndarray]]: A tuple containing:
        - cp.ndarray: Final output logits with shape (batch_size, tar_seq_len, vocab_size).
        - dict: Dictionary of attention weights from the decoder.
    """
    enc_output = self.encoder.forward(
        inp, enc_padding_mask, training
    )  # (batch_size, inp_seq_len, d_model)
    dec_output, attention_weights = self.decoder.forward(
        tar, enc_output, look_ahead_mask, dec_padding_mask, training
    )  # (batch_size, tar_seq_len, d_model)
    final_output = cp.matmul(
        dec_output, self.final_layer
    )  # (batch_size, tar_seq_len, vocab_size)
    return final_output, attention_weights

  def update_parameters(self, optimizer: SGD):
    # In a real scenario, gradients would be computed via a backward pass
    # For demonstration, we'll assume gradients are available for parameters
    # and the optimizer will use them to update.
    optimizer.step()
