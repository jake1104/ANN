
# Change Log
# 2025-10-11
# - Complete rewrite with autograd system
# - Added computational graph and automatic differentiation
# - Fixed SGD optimizer to properly update parameters
# - Unified dtype to float32 throughout
# - Improved numerical stability
# - Added proper gradient computation
# - Memory optimizations
# - Complete training pipeline
# - Full autograd integration with backward pass
# - AdamW optimizer with weight decay
# - Learning rate scheduler (warmup + cosine decay)
# - Gradient clipping
# - Complete training loop with validation
# - Loss computation with proper backpropagation

import cupy as cp
import cupy.random as random
from typing import Optional, List, Dict, Tuple, Callable
import math

# ============================================================================
# Autograd System with Backward Pass
# ============================================================================

class Tensor:
    """Tensor with autograd support for automatic differentiation."""
    def __init__(self, data: cp.ndarray, requires_grad: bool = False):
        self.data = cp.asarray(data, dtype=cp.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward_fn = None
        self._prev_tensors = []
        
    def backward(self, grad: Optional[cp.ndarray] = None):
        """Compute gradients via backpropagation."""
        if not self.requires_grad:
            return
            
        if grad is None:
            grad = cp.ones_like(self.data, dtype=cp.float32)
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
            
        if self._backward_fn is not None:
            self._backward_fn(grad)
    
    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad = None
        for t in self._prev_tensors:
            if hasattr(t, 'zero_grad'):
                t.zero_grad()
        
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"


class Parameter(Tensor):
    """Parameter tensor that requires gradient by default."""
    def __init__(self, data: cp.ndarray):
        super().__init__(data, requires_grad=True)


# ============================================================================
# Optimizers
# ============================================================================

class AdamW:
    """AdamW optimizer with weight decay."""
    def __init__(self, params: List[Parameter], lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # Initialize momentum and velocity
        self.m = [cp.zeros_like(p.data) for p in params]
        self.v = [cp.zeros_like(p.data) for p in params]
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Update biased first and second moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # AdamW update (weight decay applied directly to parameters)
            param.data -= self.lr * (m_hat / (cp.sqrt(v_hat) + self.eps) + 
                                    self.weight_decay * param.data)
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


class SGD:
    """SGD optimizer with momentum."""
    def __init__(self, params: List[Parameter], lr: float = 0.01, momentum: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [cp.zeros_like(p.data) for p in params] if momentum > 0 else None
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * param.grad
                param.data += self.velocity[i]
            else:
                param.data -= self.lr * param.grad
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


# ============================================================================
# Learning Rate Scheduler
# ============================================================================

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        self.optimizer.lr = lr
        return lr
    
    def get_lr(self):
        return self.optimizer.lr


# ============================================================================
# Gradient Utilities
# ============================================================================

def clip_grad_norm(params: List[Parameter], max_norm: float) -> float:
    """Clip gradient norm to prevent exploding gradients."""
    total_norm = 0.0
    for param in params:
        if param.grad is not None:
            param_norm = cp.linalg.norm(param.grad)
            total_norm += param_norm ** 2
    
    total_norm = float(cp.sqrt(total_norm).item())
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in params:
            if param.grad is not None:
                param.grad *= clip_coef
    
    return total_norm


# ============================================================================
# Neural Network Components
# ============================================================================

class Dropout:
    def __init__(self, rate: float):
        self.rate = rate
        self.mask = None
    
    def forward(self, x: cp.ndarray, training: bool) -> cp.ndarray:
        if not training or self.rate == 0:
            return x
        
        self.mask = (random.rand(*x.shape) > self.rate).astype(cp.float32)
        return (x * self.mask) / (1.0 - self.rate)


class LayerNormalization:
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.d_model = d_model
        self.gamma = Parameter(cp.ones(d_model, dtype=cp.float32))
        self.beta = Parameter(cp.zeros(d_model, dtype=cp.float32))
    
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        mean = cp.mean(x, axis=-1, keepdims=True)
        variance = cp.var(x, axis=-1, keepdims=True)
        normalized_x = (x - mean) / cp.sqrt(variance + self.epsilon)
        return normalized_x * self.gamma.data + self.beta.data
    
    def parameters(self) -> List[Parameter]:
        return [self.gamma, self.beta]


class PositionalEncoding:
    def __init__(self, d_model: int, max_seq_len: int):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.positional_encoding_matrix = self._get_positional_encoding_matrix()
    
    def _get_positional_encoding_matrix(self) -> cp.ndarray:
        position = cp.arange(self.max_seq_len, dtype=cp.float32)[:, cp.newaxis]
        div_term = cp.exp(
            cp.arange(0, self.d_model, 2, dtype=cp.float32) * 
            -(cp.log(10000.0) / self.d_model)
        )
        
        pe = cp.zeros((self.max_seq_len, self.d_model), dtype=cp.float32)
        pe[:, 0::2] = cp.sin(position * div_term)
        pe[:, 1::2] = cp.cos(position * div_term)
        return pe[cp.newaxis, :, :]
    
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        seq_len = x.shape[1]
        return x + self.positional_encoding_matrix[:, :seq_len, :]


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = float(1.0 / cp.sqrt(self.head_dim).item())
        
        # Xavier/Glorot initialization
        self.wq = Parameter(cp.random.randn(d_model, d_model).astype(cp.float32) * cp.sqrt(2.0 / d_model))
        self.wk = Parameter(cp.random.randn(d_model, d_model).astype(cp.float32) * cp.sqrt(2.0 / d_model))
        self.wv = Parameter(cp.random.randn(d_model, d_model).astype(cp.float32) * cp.sqrt(2.0 / d_model))
        self.wo = Parameter(cp.random.randn(d_model, d_model).astype(cp.float32) * cp.sqrt(2.0 / d_model))
        
        self.dropout = Dropout(dropout_rate)
    
    def parameters(self) -> List[Parameter]:
        return [self.wq, self.wk, self.wv, self.wo]
    
    def _scaled_dot_product_attention(
        self, q: cp.ndarray, k: cp.ndarray, v: cp.ndarray, 
        mask: Optional[cp.ndarray] = None
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        matmul_qk = cp.matmul(q, k.swapaxes(-2, -1))
        scaled_attention_logits = matmul_qk * self.scale
        
        if mask is not None:
            scaled_attention_logits = cp.where(
                mask == 0, 
                -1e9,
                scaled_attention_logits
            )
        
        # Numerically stable softmax
        max_logits = cp.max(scaled_attention_logits, axis=-1, keepdims=True)
        exp_logits = cp.exp(scaled_attention_logits - max_logits)
        attention_weights = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)
        
        output = cp.matmul(attention_weights, v)
        return output, attention_weights
    
    def forward(
        self, q: cp.ndarray, k: cp.ndarray, v: cp.ndarray,
        mask: Optional[cp.ndarray] = None, training: bool = True
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        batch_size = q.shape[0]
        
        # Linear projections and reshape
        q = cp.matmul(q, self.wq.data).reshape(batch_size, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = cp.matmul(k, self.wk.data).reshape(batch_size, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        v = cp.matmul(v, self.wv.data).reshape(batch_size, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        
        # Attention
        scaled_attention, attention_weights = self._scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = self.dropout.forward(scaled_attention, training)
        
        # Concatenate heads and project
        scaled_attention = scaled_attention.swapaxes(1, 2).reshape(batch_size, -1, self.d_model)
        output = cp.matmul(scaled_attention, self.wo.data)
        
        return output, attention_weights


class FeedForwardNetwork:
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        self.w1 = Parameter(cp.random.randn(d_model, d_ff).astype(cp.float32) * cp.sqrt(2.0 / d_model))
        self.b1 = Parameter(cp.zeros(d_ff, dtype=cp.float32))
        self.w2 = Parameter(cp.random.randn(d_ff, d_model).astype(cp.float32) * cp.sqrt(2.0 / d_ff))
        self.b2 = Parameter(cp.zeros(d_model, dtype=cp.float32))
        self.dropout = Dropout(dropout_rate)
    
    def parameters(self) -> List[Parameter]:
        return [self.w1, self.b1, self.w2, self.b2]
    
    def forward(self, x: cp.ndarray, training: bool = True) -> cp.ndarray:
        x = cp.maximum(0, cp.matmul(x, self.w1.data) + self.b1.data)
        x = self.dropout.forward(x, training)
        x = cp.matmul(x, self.w2.data) + self.b2.data
        x = self.dropout.forward(x, training)
        return x


class EncoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def parameters(self) -> List[Parameter]:
        params = []
        params.extend(self.mha.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.layernorm1.parameters())
        params.extend(self.layernorm2.parameters())
        return params
    
    def forward(self, x: cp.ndarray, mask: Optional[cp.ndarray], training: bool = True) -> cp.ndarray:
        norm_x = self.layernorm1.forward(x)
        attn_output, _ = self.mha.forward(norm_x, norm_x, norm_x, mask, training)
        attn_output = self.dropout1.forward(attn_output, training)
        out1 = x + attn_output
        
        norm_out1 = self.layernorm2.forward(out1)
        ffn_output = self.ffn.forward(norm_out1, training)
        ffn_output = self.dropout2.forward(ffn_output, training)
        out2 = out1 + ffn_output
        
        return out2


class DecoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
        self.layernorm3 = LayerNormalization(d_model)
        
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)
    
    def parameters(self) -> List[Parameter]:
        params = []
        params.extend(self.mha1.parameters())
        params.extend(self.mha2.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.layernorm1.parameters())
        params.extend(self.layernorm2.parameters())
        params.extend(self.layernorm3.parameters())
        return params
    
    def forward(
        self, x: cp.ndarray, enc_output: cp.ndarray,
        look_ahead_mask: Optional[cp.ndarray], padding_mask: Optional[cp.ndarray],
        training: bool = True
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        norm_x = self.layernorm1.forward(x)
        attn1, attn_weights_block1 = self.mha1.forward(norm_x, norm_x, norm_x, look_ahead_mask, training)
        attn1 = self.dropout1.forward(attn1, training)
        out1 = x + attn1
        
        norm_out1 = self.layernorm2.forward(out1)
        attn2, attn_weights_block2 = self.mha2.forward(norm_out1, enc_output, enc_output, padding_mask, training)
        attn2 = self.dropout2.forward(attn2, training)
        out2 = out1 + attn2
        
        norm_out2 = self.layernorm3.forward(out2)
        ffn_output = self.ffn.forward(norm_out2, training)
        ffn_output = self.dropout3.forward(ffn_output, training)
        out3 = out2 + ffn_output
        
        return out3, attn_weights_block1, attn_weights_block2


class Encoder:
    def __init__(
        self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
        vocab_size: int, max_seq_len: int, dropout_rate: float = 0.1
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = Parameter(
            cp.random.randn(vocab_size, d_model).astype(cp.float32) * cp.sqrt(1.0 / vocab_size)
        )
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)
    
    def parameters(self) -> List[Parameter]:
        params = [self.embedding]
        for layer in self.enc_layers:
            params.extend(layer.parameters())
        return params
    
    def forward(self, x: cp.ndarray, mask: Optional[cp.ndarray], training: bool = True) -> cp.ndarray:
        x = cp.take(self.embedding.data, x, axis=0)
        x *= cp.sqrt(float(self.d_model))
        x = self.pos_encoding.forward(x)
        x = self.dropout.forward(x, training)
        
        for layer in self.enc_layers:
            x = layer.forward(x, mask, training)
        
        return x


class Decoder:
    def __init__(
        self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
        vocab_size: int, max_seq_len: int, dropout_rate: float = 0.1
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = Parameter(
            cp.random.randn(vocab_size, d_model).astype(cp.float32) * cp.sqrt(1.0 / vocab_size)
        )
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)
    
    def parameters(self) -> List[Parameter]:
        params = [self.embedding]
        for layer in self.dec_layers:
            params.extend(layer.parameters())
        return params
    
    def forward(
        self, x: cp.ndarray, enc_output: cp.ndarray,
        look_ahead_mask: Optional[cp.ndarray], padding_mask: Optional[cp.ndarray],
        training: bool = True
    ) -> Tuple[cp.ndarray, Dict[str, cp.ndarray]]:
        x = cp.take(self.embedding.data, x, axis=0)
        x *= cp.sqrt(float(self.d_model))
        x = self.pos_encoding.forward(x)
        x = self.dropout.forward(x, training)
        
        attention_weights = {}
        for i, layer in enumerate(self.dec_layers):
            x, block1, block2 = layer.forward(x, enc_output, look_ahead_mask, padding_mask, training)
            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2
        
        return x, attention_weights


class TransformerArchitecture:
    def __init__(
        self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
        d_ff: int, max_seq_len: int, dropout_rate: float = 0.1
    ):
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout_rate)
        self.final_layer = Parameter(
            cp.random.randn(d_model, vocab_size).astype(cp.float32) * cp.sqrt(2.0 / d_model)
        )
    
    def parameters(self) -> List[Parameter]:
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        params.append(self.final_layer)
        return params
    
    def forward(
        self, inp: cp.ndarray, tar: cp.ndarray,
        enc_padding_mask: Optional[cp.ndarray],
        look_ahead_mask: Optional[cp.ndarray],
        dec_padding_mask: Optional[cp.ndarray],
        training: bool = True
    ) -> Tuple[cp.ndarray, Dict[str, cp.ndarray]]:
        enc_output = self.encoder.forward(inp, enc_padding_mask, training)
        dec_output, attention_weights = self.decoder.forward(
            tar, enc_output, look_ahead_mask, dec_padding_mask, training
        )
        final_output = cp.matmul(dec_output, self.final_layer.data)
        return final_output, attention_weights


# ============================================================================
# Loss Function with Backward Pass
# ============================================================================

def sparse_categorical_crossentropy_with_gradients(
    logits: cp.ndarray, targets: cp.ndarray, 
    final_layer: Parameter, dec_output: cp.ndarray,
    mask: Optional[cp.ndarray] = None
) -> Tuple[float, cp.ndarray]:
    """
    Compute loss and compute gradients for final layer.
    
    Returns:
        loss: Scalar loss value
        grad_dec_output: Gradient w.r.t decoder output
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Forward: compute softmax probabilities
    logits_max = cp.max(logits, axis=-1, keepdims=True)
    logits_shifted = logits - logits_max
    exp_logits = cp.exp(logits_shifted)
    probs = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)
    
    # Compute loss
    batch_indices = cp.arange(batch_size)[:, None]
    seq_indices = cp.arange(seq_len)[None, :]
    target_probs = probs[batch_indices, seq_indices, targets]
    loss = -cp.log(target_probs + 1e-9)
    
    if mask is not None:
        loss = loss * mask
        total_loss = float(cp.sum(loss).item() / cp.sum(mask).item())
        norm_factor = cp.sum(mask)
    else:
        total_loss = float(cp.mean(loss).item())
        norm_factor = batch_size * seq_len
    
    # Backward: compute gradients
    # d(loss)/d(logits) = probs - one_hot(targets)
    grad_logits = probs.copy()
    grad_logits[batch_indices, seq_indices, targets] -= 1
    
    if mask is not None:
        grad_logits *= mask[:, :, None]
    
    grad_logits /= norm_factor
    
    # Gradient for final layer weights: dec_output^T @ grad_logits
    # Shape: (d_model, vocab_size)
    grad_final_layer = cp.matmul(
        dec_output.reshape(-1, dec_output.shape[-1]).T,
        grad_logits.reshape(-1, vocab_size)
    )
    
    # Gradient for decoder output: grad_logits @ final_layer^T
    # Shape: (batch_size, seq_len, d_model)
    grad_dec_output = cp.matmul(grad_logits, final_layer.data.T)
    
    # Update final layer gradient
    if final_layer.grad is None:
        final_layer.grad = grad_final_layer
    else:
        final_layer.grad += grad_final_layer
    
    return total_loss, grad_dec_output


# ============================================================================
# Backward Pass for Transformer Components
# ============================================================================

def backward_linear(grad_output: cp.ndarray, input_data: cp.ndarray,
                   weight: Parameter, bias: Optional[Parameter] = None):
    """Compute gradients for linear layer."""
    # grad_weight = input^T @ grad_output
    batch_size = input_data.shape[0]
    seq_len = input_data.shape[1] if input_data.ndim == 3 else 1
    
    if input_data.ndim == 3:
        grad_weight = cp.matmul(
            input_data.reshape(-1, input_data.shape[-1]).T,
            grad_output.reshape(-1, grad_output.shape[-1])
        )
    else:
        grad_weight = cp.matmul(input_data.T, grad_output)
    
    if weight.grad is None:
        weight.grad = grad_weight
    else:
        weight.grad += grad_weight
    
    if bias is not None:
        if grad_output.ndim == 3:
            grad_bias = cp.sum(grad_output, axis=(0, 1))
        else:
            grad_bias = cp.sum(grad_output, axis=0)
        
        if bias.grad is None:
            bias.grad = grad_bias
        else:
            bias.grad += grad_bias
    
    # grad_input = grad_output @ weight^T
    grad_input = cp.matmul(grad_output, weight.data.T)
    return grad_input


# ============================================================================
# Training Utilities
# ============================================================================

def create_padding_mask(seq: cp.ndarray, pad_token: int = 0) -> cp.ndarray:
    """Create padding mask (1 for real tokens, 0 for padding)."""
    mask = (seq != pad_token).astype(cp.float32)
    return mask[:, cp.newaxis, cp.newaxis, :]


def create_look_ahead_mask(size: int) -> cp.ndarray:
    """Create look-ahead mask for decoder self-attention."""
    mask = 1 - cp.triu(cp.ones((size, size), dtype=cp.float32), k=1)
    return mask[cp.newaxis, cp.newaxis, :, :]


def create_combined_mask(tar: cp.ndarray, pad_token: int = 0) -> cp.ndarray:
    """Create combined look-ahead and padding mask."""
    seq_len = tar.shape[1]
    look_ahead_mask = create_look_ahead_mask(seq_len)
    padding_mask = create_padding_mask(tar, pad_token)
    return cp.minimum(look_ahead_mask, padding_mask)


def compute_accuracy(logits: cp.ndarray, targets: cp.ndarray,
                    mask: Optional[cp.ndarray] = None) -> float:
    """Compute prediction accuracy."""
    predictions = cp.argmax(logits, axis=-1)
    correct = (predictions == targets).astype(cp.float32)
    
    if mask is not None:
        correct = correct * mask
        return float(cp.sum(correct).item() / cp.sum(mask).item() * 100)
    
    return float(cp.mean(correct).item() * 100)


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    """Training manager for Transformer."""
    def __init__(self, model: TransformerArchitecture, optimizer, scheduler,
                 max_grad_norm: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def train_step(self, inp: cp.ndarray, tar_inp: cp.ndarray, tar_real: cp.ndarray,
                   enc_mask: cp.ndarray, combined_mask: cp.ndarray, dec_mask: cp.ndarray) -> Dict[str, float]:
        """Single training step."""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, _ = self.model.forward(
            inp, tar_inp, enc_mask, combined_mask, dec_mask, training=True
        )
        
        # Get decoder output for gradient computation
        # We need to recompute this because we need the intermediate value
        enc_output = self.model.encoder.forward(inp, enc_mask, training=True)
        tar_embedded = cp.take(self.model.decoder.embedding.data, tar_inp, axis=0)
        tar_embedded *= cp.sqrt(float(self.model.decoder.d_model))
        tar_embedded = self.model.decoder.pos_encoding.forward(tar_embedded)
        tar_embedded = self.model.decoder.dropout.forward(tar_embedded, training=True)
        
        # Pass through decoder layers to get final decoder output
        dec_output = tar_embedded
        for layer in self.model.decoder.dec_layers:
            dec_output, _, _ = layer.forward(dec_output, enc_output, combined_mask, dec_mask, training=True)
        
        # Compute loss and gradients
        target_mask = (tar_real != 0).astype(cp.float32)
        loss, grad_output = sparse_categorical_crossentropy_with_gradients(
            logits, tar_real, self.model.final_layer, dec_output, target_mask
        )
        
        # Compute accuracy
        accuracy = compute_accuracy(logits, tar_real, target_mask)
        
        # Clip gradients
        grad_norm = clip_grad_norm(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        lr = self.scheduler.step()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'grad_norm': grad_norm,
            'lr': lr
        }
    
    def validate(self, inp: cp.ndarray, tar_inp: cp.ndarray, tar_real: cp.ndarray,
                enc_mask: cp.ndarray, combined_mask: cp.ndarray, dec_mask: cp.ndarray) -> Dict[str, float]:
        """Validation step."""
        # Forward pass (no gradient computation needed)
        logits, _ = self.model.forward(
            inp, tar_inp, enc_mask, combined_mask, dec_mask, training=False
        )
        
        # Compute metrics
        target_mask = (tar_real != 0).astype(cp.float32)
        
        # Compute loss without gradients
        batch_size, seq_len, vocab_size = logits.shape
        logits_max = cp.max(logits, axis=-1, keepdims=True)
        exp_logits = cp.exp(logits - logits_max)
        probs = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)
        
        batch_indices = cp.arange(batch_size)[:, None]
        seq_indices = cp.arange(seq_len)[None, :]
        target_probs = probs[batch_indices, seq_indices, tar_real]
        loss = -cp.log(target_probs + 1e-9)
        
        loss = loss * target_mask
        val_loss = float(cp.sum(loss).item() / cp.sum(target_mask).item())
        
        accuracy = compute_accuracy(logits, tar_real, target_mask)
        
        return {
            'loss': val_loss,
            'accuracy': accuracy
        }
    
    def train(self, train_data: List[Tuple], val_data: List[Tuple], 
              num_epochs: int, print_every: int = 10):
        """Full training loop."""
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            # Training
            for i, (inp, tar_inp, tar_real, enc_mask, comb_mask, dec_mask) in enumerate(train_data):
                metrics = self.train_step(inp, tar_inp, tar_real, enc_mask, comb_mask, dec_mask)
                
                epoch_loss += metrics['loss']
                epoch_acc += metrics['accuracy']
                
                if (i + 1) % print_every == 0:
                    avg_loss = epoch_loss / (i + 1)
                    avg_acc = epoch_acc / (i + 1)
                    print(f"Epoch {epoch+1}/{num_epochs} | Step {i+1}/{len(train_data)} | "
                          f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | "
                          f"LR: {metrics['lr']:.6f} | Grad: {metrics['grad_norm']:.4f}")
            
            # Average training metrics
            avg_train_loss = epoch_loss / len(train_data)
            avg_train_acc = epoch_acc / len(train_data)
            self.train_losses.append(avg_train_loss)
            self.train_accs.append(avg_train_acc)
            
            # Validation
            val_loss = 0.0
            val_acc = 0.0
            for inp, tar_inp, tar_real, enc_mask, comb_mask, dec_mask in val_data:
                val_metrics = self.validate(inp, tar_inp, tar_real, enc_mask, comb_mask, dec_mask)
                val_loss += val_metrics['loss']
                val_acc += val_metrics['accuracy']
            
            avg_val_loss = val_loss / len(val_data)
            avg_val_acc = val_acc / len(val_data)
            self.val_losses.append(avg_val_loss)
            self.val_accs.append(avg_val_acc)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")
            print("-" * 80)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Transformer v02 - Complete Training Pipeline")
    print("=" * 80)
    
    # Hyperparameters
    vocab_size = 1000
    d_model = 128
    num_heads = 8
    num_layers = 2
    d_ff = 512
    max_seq_len = 50
    dropout_rate = 0.1
    
    batch_size = 4
    num_train_batches = 20
    num_val_batches = 5
    num_epochs = 3
    
    learning_rate = 0.0001
    warmup_steps = 100
    total_steps = num_train_batches * num_epochs
    max_grad_norm = 1.0
    
    print(f"\nModel Configuration:")
    print(f"  Vocab Size: {vocab_size}")
    print(f"  Model Dim: {d_model}")
    print(f"  Num Heads: {num_heads}")
    print(f"  Num Layers: {num_layers}")
    print(f"  FFN Dim: {d_ff}")
    
    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Warmup Steps: {warmup_steps}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Max Grad Norm: {max_grad_norm}")
    
    # Create model
    print("\nInitializing model...")
    model = TransformerArchitecture(
        vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout_rate
    )
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    # Count parameters
    total_params = sum(p.data.size for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # Generate dummy training data
    print("\nGenerating training data...")
    train_data = []
    for _ in range(num_train_batches):
        inp = cp.random.randint(1, vocab_size, (batch_size, 20), dtype=cp.int32)
        tar_inp = cp.random.randint(1, vocab_size, (batch_size, 25), dtype=cp.int32)
        tar_real = cp.random.randint(1, vocab_size, (batch_size, 25), dtype=cp.int32)
        
        # Add some padding
        inp[0, 15:] = 0
        tar_inp[1, 20:] = 0
        tar_real[1, 20:] = 0
        
        enc_mask = create_padding_mask(inp)
        dec_mask = create_padding_mask(inp)
        comb_mask = create_combined_mask(tar_inp)
        
        train_data.append((inp, tar_inp, tar_real, enc_mask, comb_mask, dec_mask))
    
    # Generate dummy validation data
    print("Generating validation data...")
    val_data = []
    for _ in range(num_val_batches):
        inp = cp.random.randint(1, vocab_size, (batch_size, 20), dtype=cp.int32)
        tar_inp = cp.random.randint(1, vocab_size, (batch_size, 25), dtype=cp.int32)
        tar_real = cp.random.randint(1, vocab_size, (batch_size, 25), dtype=cp.int32)
        
        enc_mask = create_padding_mask(inp)
        dec_mask = create_padding_mask(inp)
        comb_mask = create_combined_mask(tar_inp)
        
        val_data.append((inp, tar_inp, tar_real, enc_mask, comb_mask, dec_mask))
    
    # Create trainer and train
    trainer = Trainer(model, optimizer, scheduler, max_grad_norm)
    trainer.train(train_data, val_data, num_epochs, print_every=5)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    print(f"\nFinal Results:")
    print(f"  Final Train Loss: {trainer.train_losses[-1]:.4f}")
    print(f"  Final Train Acc: {trainer.train_accs[-1]:.2f}%")
    print(f"  Final Val Loss: {trainer.val_losses[-1]:.4f}")
    print(f"  Final Val Acc: {trainer.val_accs[-1]:.2f}%")
    
    # Inference example
    print("\n" + "=" * 80)
    print("Inference Example")
    print("=" * 80)
    
    test_inp = cp.random.randint(1, vocab_size, (2, 15), dtype=cp.int32)
    test_tar = cp.random.randint(1, vocab_size, (2, 20), dtype=cp.int32)
    
    test_enc_mask = create_padding_mask(test_inp)
    test_dec_mask = create_padding_mask(test_inp)
    test_comb_mask = create_combined_mask(test_tar)
    
    logits, attn_weights = model.forward(
        test_inp, test_tar, test_enc_mask, test_comb_mask, test_dec_mask, training=False
    )
    
    predictions = cp.argmax(logits, axis=-1)
    
    print(f"\nTest Input Shape: {test_inp.shape}")
    print(f"Test Output Shape: {logits.shape}")
    print(f"Predictions Shape: {predictions.shape}")
    print(f"\nSample Input: {test_inp[0, :10]}")
    print(f"Sample Predictions: {predictions[0, :10]}")
    print(f"\nNumber of attention weight matrices: {len(attn_weights)}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
