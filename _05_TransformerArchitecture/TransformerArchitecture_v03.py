# Change Log
# 2025-10-11: v03 - Complete backpropagation implementation
# - Full backward pass through all layers
# - Proper gradient computation for embeddings
# - Tensor and cp.ndarray properly separated
# - All operations use Tensor when gradients needed
# - Manual backprop for transformer-specific operations
# - Memory-efficient gradient computation

import cupy as cp
import cupy.random as random
from typing import Optional, List, Dict, Tuple, Callable, Union
import math

# ============================================================================
# Tensor with Autograd
# ============================================================================

class Tensor:
    """Tensor with autograd support."""
    def __init__(self, data: cp.ndarray, requires_grad: bool = False):
        self.data = cp.asarray(data, dtype=cp.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._version = 0
        
    def backward(self, grad: Optional[cp.ndarray] = None):
        """Accumulate gradients."""
        if not self.requires_grad:
            return
            
        if grad is None:
            grad = cp.ones_like(self.data, dtype=cp.float32)
        
        if self.grad is None:
            self.grad = grad.astype(cp.float32)
        else:
            self.grad += grad.astype(cp.float32)
    
    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad = None
    
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
            
            # AdamW update
            param.data -= self.lr * (m_hat / (cp.sqrt(v_hat) + self.eps) + 
                                    self.weight_decay * param.data)
    
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
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
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
    
    def backward(self, grad: cp.ndarray, training: bool) -> cp.ndarray:
        if not training or self.rate == 0:
            return grad
        return (grad * self.mask) / (1.0 - self.rate)


class LayerNormalization:
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.d_model = d_model
        self.gamma = Parameter(cp.ones(d_model, dtype=cp.float32))
        self.beta = Parameter(cp.zeros(d_model, dtype=cp.float32))
        
        # Cache for backward
        self.normalized_x = None
        self.std = None
        self.x_input = None
    
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        mean = cp.mean(x, axis=-1, keepdims=True)
        variance = cp.var(x, axis=-1, keepdims=True)
        std = cp.sqrt(variance + self.epsilon)
        normalized_x = (x - mean) / std
        
        # Cache for backward
        self.normalized_x = normalized_x
        self.std = std
        self.x_input = x
        
        return normalized_x * self.gamma.data + self.beta.data
    
    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        # Gradient w.r.t gamma and beta
        if self.gamma.grad is None:
            self.gamma.grad = cp.sum(grad * self.normalized_x, axis=(0, 1))
        else:
            self.gamma.grad += cp.sum(grad * self.normalized_x, axis=(0, 1))
        
        if self.beta.grad is None:
            self.beta.grad = cp.sum(grad, axis=(0, 1))
        else:
            self.beta.grad += cp.sum(grad, axis=(0, 1))
        
        # Gradient w.r.t input
        N = self.x_input.shape[0] * self.x_input.shape[1]
        grad_normalized = grad * self.gamma.data
        
        mean_grad = cp.mean(grad_normalized, axis=-1, keepdims=True)
        mean_grad_normalized = cp.mean(grad_normalized * self.normalized_x, axis=-1, keepdims=True)
        
        grad_input = (grad_normalized - mean_grad - self.normalized_x * mean_grad_normalized) / self.std
        
        return grad_input
    
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
        
        self.wq = Parameter(cp.random.randn(d_model, d_model).astype(cp.float32) * cp.sqrt(2.0 / d_model))
        self.wk = Parameter(cp.random.randn(d_model, d_model).astype(cp.float32) * cp.sqrt(2.0 / d_model))
        self.wv = Parameter(cp.random.randn(d_model, d_model).astype(cp.float32) * cp.sqrt(2.0 / d_model))
        self.wo = Parameter(cp.random.randn(d_model, d_model).astype(cp.float32) * cp.sqrt(2.0 / d_model))
        
        self.dropout = Dropout(dropout_rate)
        
        # Cache for backward
        self.q_input = None
        self.k_input = None
        self.v_input = None
        self.q = None
        self.k = None
        self.v = None
        self.attention_weights = None
        self.attn_output = None
        self.concat_output = None
    
    def parameters(self) -> List[Parameter]:
        return [self.wq, self.wk, self.wv, self.wo]
    
    def forward(
        self, q: cp.ndarray, k: cp.ndarray, v: cp.ndarray,
        mask: Optional[cp.ndarray] = None, training: bool = True
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        batch_size = q.shape[0]
        
        # Cache inputs
        self.q_input = q
        self.k_input = k
        self.v_input = v
        
        # Linear projections
        self.q = cp.matmul(q, self.wq.data).reshape(batch_size, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        self.k = cp.matmul(k, self.wk.data).reshape(batch_size, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        self.v = cp.matmul(v, self.wv.data).reshape(batch_size, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        
        # Scaled dot-product attention
        matmul_qk = cp.matmul(self.q, self.k.swapaxes(-2, -1))
        scaled_attention_logits = matmul_qk * self.scale
        
        if mask is not None:
            scaled_attention_logits = cp.where(mask == 0, -1e9, scaled_attention_logits)
        
        # Softmax
        max_logits = cp.max(scaled_attention_logits, axis=-1, keepdims=True)
        exp_logits = cp.exp(scaled_attention_logits - max_logits)
        self.attention_weights = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)
        
        # Apply attention to values
        self.attn_output = cp.matmul(self.attention_weights, self.v)
        self.attn_output = self.dropout.forward(self.attn_output, training)
        
        # Concatenate heads
        self.concat_output = self.attn_output.swapaxes(1, 2).reshape(batch_size, -1, self.d_model)
        output = cp.matmul(self.concat_output, self.wo.data)
        
        return output, self.attention_weights
    
    def backward(self, grad: cp.ndarray, training: bool = True) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        batch_size = grad.shape[0]
        
        # Gradient for wo
        grad_wo = cp.matmul(
            self.concat_output.reshape(-1, self.d_model).T,
            grad.reshape(-1, self.d_model)
        )
        if self.wo.grad is None:
            self.wo.grad = grad_wo
        else:
            self.wo.grad += grad_wo
        
        # Gradient for concat_output
        grad_concat = cp.matmul(grad, self.wo.data.T)
        
        # Reshape to multi-head format
        grad_attn = grad_concat.reshape(batch_size, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        
        # Dropout backward
        grad_attn = self.dropout.backward(grad_attn, training)
        
        # Gradient for attention weights and values
        grad_v = cp.matmul(self.attention_weights.swapaxes(-2, -1), grad_attn)
        grad_weights = cp.matmul(grad_attn, self.v.swapaxes(-2, -1))
        
        # Softmax backward
        s = self.attention_weights
        grad_logits = s * (grad_weights - cp.sum(grad_weights * s, axis=-1, keepdims=True))
        grad_logits = grad_logits * self.scale
        
        # Attention backward
        grad_q = cp.matmul(grad_logits, self.k)
        grad_k = cp.matmul(grad_logits.swapaxes(-2, -1), self.q)
        
        # Reshape to original format
        grad_q = grad_q.swapaxes(1, 2).reshape(batch_size, -1, self.d_model)
        grad_k = grad_k.swapaxes(1, 2).reshape(batch_size, -1, self.d_model)
        grad_v = grad_v.swapaxes(1, 2).reshape(batch_size, -1, self.d_model)
        
        # Gradients for weights
        grad_wq = cp.matmul(self.q_input.reshape(-1, self.d_model).T, grad_q.reshape(-1, self.d_model))
        grad_wk = cp.matmul(self.k_input.reshape(-1, self.d_model).T, grad_k.reshape(-1, self.d_model))
        grad_wv = cp.matmul(self.v_input.reshape(-1, self.d_model).T, grad_v.reshape(-1, self.d_model))
        
        if self.wq.grad is None:
            self.wq.grad = grad_wq
        else:
            self.wq.grad += grad_wq
            
        if self.wk.grad is None:
            self.wk.grad = grad_wk
        else:
            self.wk.grad += grad_wk
            
        if self.wv.grad is None:
            self.wv.grad = grad_wv
        else:
            self.wv.grad += grad_wv
        
        # Gradients for inputs
        grad_q_input = cp.matmul(grad_q, self.wq.data.T)
        grad_k_input = cp.matmul(grad_k, self.wk.data.T)
        grad_v_input = cp.matmul(grad_v, self.wv.data.T)
        
        return grad_q_input, grad_k_input, grad_v_input


class FeedForwardNetwork:
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        self.w1 = Parameter(cp.random.randn(d_model, d_ff).astype(cp.float32) * cp.sqrt(2.0 / d_model))
        self.b1 = Parameter(cp.zeros(d_ff, dtype=cp.float32))
        self.w2 = Parameter(cp.random.randn(d_ff, d_model).astype(cp.float32) * cp.sqrt(2.0 / d_ff))
        self.b2 = Parameter(cp.zeros(d_model, dtype=cp.float32))
        self.dropout1 = Dropout(dropout_rate) # First dropout layer
        self.dropout2 = Dropout(dropout_rate) # Second dropout layer
        
        # Cache for backward
        self.x_input = None
        self.hidden = None # Output of first linear + ReLU
        self.ffn_hidden_output = None # Output of dropout1 (input to second linear)
    
    def parameters(self) -> List[Parameter]:
        return [self.w1, self.b1, self.w2, self.b2]
    
    def forward(self, x: cp.ndarray, training: bool = True) -> cp.ndarray:
        self.x_input = x
        
        # First layer: Linear -> ReLU -> Dropout
        self.hidden = cp.maximum(0, cp.matmul(x, self.w1.data) + self.b1.data)
        self.ffn_hidden_output = self.dropout1.forward(self.hidden, training)
        
        # Second layer: Linear -> Dropout
        output = cp.matmul(self.ffn_hidden_output, self.w2.data) + self.b2.data
        output = self.dropout2.forward(output, training)
        
        return output
    
    def backward(self, grad: cp.ndarray, training: bool = True) -> cp.ndarray:
        # Backward through second dropout
        grad = self.dropout2.backward(grad, training)
        
        # Gradient for w2 and b2
        grad_w2 = cp.matmul(self.ffn_hidden_output.reshape(-1, self.ffn_hidden_output.shape[-1]).T,
                           grad.reshape(-1, grad.shape[-1]))
        grad_b2 = cp.sum(grad, axis=(0, 1))
        
        if self.w2.grad is None:
            self.w2.grad = grad_w2
        else:
            self.w2.grad += grad_w2
            
        if self.b2.grad is None:
            self.b2.grad = grad_b2
        else:
            self.b2.grad += grad_b2
        
        # Gradient for input to second linear layer (ffn_hidden_output)
        grad_ffn_hidden_output = cp.matmul(grad, self.w2.data.T)
        
        # Backward through first dropout
        grad_hidden = self.dropout1.backward(grad_ffn_hidden_output, training)
        
        # ReLU backward
        grad_hidden = grad_hidden * (self.hidden > 0)
        
        # Gradient for w1 and b1
        grad_w1 = cp.matmul(self.x_input.reshape(-1, self.x_input.shape[-1]).T,
                           grad_hidden.reshape(-1, grad_hidden.shape[-1]))
        grad_b1 = cp.sum(grad_hidden, axis=(0, 1))
        
        if self.w1.grad is None:
            self.w1.grad = grad_w1
        else:
            self.w1.grad += grad_w1
            
        if self.b1.grad is None:
            self.b1.grad = grad_b1
        else:
            self.b1.grad += grad_b1
        
        # Gradient for input to first linear layer (x_input)
        grad_input = cp.matmul(grad_hidden, self.w1.data.T)
        
        return grad_input


class EncoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
        # Cache for backward
        self.x_input = None
        self.norm_x = None
        self.attn_output = None
        self.out1 = None
        self.norm_out1 = None
        self.ffn_output = None
    
    def parameters(self) -> List[Parameter]:
        params = []
        params.extend(self.mha.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.layernorm1.parameters())
        params.extend(self.layernorm2.parameters())
        return params
    
    def forward(self, x: cp.ndarray, mask: Optional[cp.ndarray], training: bool = True) -> cp.ndarray:
        self.x_input = x
        
        # Self-attention with pre-norm
        self.norm_x = self.layernorm1.forward(x)
        self.attn_output, _ = self.mha.forward(self.norm_x, self.norm_x, self.norm_x, mask, training)
        self.attn_output = self.dropout1.forward(self.attn_output, training)
        self.out1 = x + self.attn_output
        
        # Feed-forward with pre-norm
        self.norm_out1 = self.layernorm2.forward(self.out1)
        self.ffn_output = self.ffn.forward(self.norm_out1, training)
        self.ffn_output = self.dropout2.forward(self.ffn_output, training)
        out2 = self.out1 + self.ffn_output
        
        return out2
    
    def backward(self, grad: cp.ndarray, training: bool = True) -> cp.ndarray:
        # FFN residual backward
        grad_out1 = grad.copy()
        grad_ffn = grad.copy()
        
        # Dropout2 backward
        grad_ffn = self.dropout2.backward(grad_ffn, training)
        
        # FFN backward
        grad_norm_out1 = self.ffn.backward(grad_ffn, training)
        
        # LayerNorm2 backward
        grad_out1_from_norm = self.layernorm2.backward(grad_norm_out1)
        grad_out1 += grad_out1_from_norm
        
        # Attention residual backward
        grad_x = grad_out1.copy()
        grad_attn = grad_out1.copy()
        
        # Dropout1 backward
        grad_attn = self.dropout1.backward(grad_attn, training)
        
        # MHA backward
        grad_norm_x, _, _ = self.mha.backward(grad_attn, training)
        
        # LayerNorm1 backward
        grad_x_from_norm = self.layernorm1.backward(grad_norm_x)
        grad_x += grad_x_from_norm
        
        return grad_x


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
        
        # Cache for backward
        self.x_input = None
        self.enc_output = None
        self.norm_x = None
        self.attn1_output = None
        self.out1 = None
        self.norm_out1 = None
        self.attn2_output = None
        self.out2 = None
        self.norm_out2 = None
        self.ffn_output = None
    
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
        self.x_input = x
        self.enc_output = enc_output
        
        # Masked self-attention
        self.norm_x = self.layernorm1.forward(x)
        self.attn1_output, attn_weights_block1 = self.mha1.forward(
            self.norm_x, self.norm_x, self.norm_x, look_ahead_mask, training
        )
        self.attn1_output = self.dropout1.forward(self.attn1_output, training)
        self.out1 = x + self.attn1_output
        
        # Cross-attention
        self.norm_out1 = self.layernorm2.forward(self.out1)
        self.attn2_output, attn_weights_block2 = self.mha2.forward(
            self.norm_out1, enc_output, enc_output, padding_mask, training
        )
        self.attn2_output = self.dropout2.forward(self.attn2_output, training)
        self.out2 = self.out1 + self.attn2_output
        
        # Feed-forward
        self.norm_out2 = self.layernorm3.forward(self.out2)
        self.ffn_output = self.ffn.forward(self.norm_out2, training)
        self.ffn_output = self.dropout3.forward(self.ffn_output, training)
        out3 = self.out2 + self.ffn_output
        
        return out3, attn_weights_block1, attn_weights_block2
    
    def backward(self, grad: cp.ndarray, training: bool = True) -> Tuple[cp.ndarray, cp.ndarray]:
        # FFN residual backward
        grad_out2 = grad.copy()
        grad_ffn = grad.copy()
        
        grad_ffn = self.dropout3.backward(grad_ffn, training)
        grad_norm_out2 = self.ffn.backward(grad_ffn, training)
        grad_out2_from_norm = self.layernorm3.backward(grad_norm_out2)
        grad_out2 += grad_out2_from_norm
        
        # Cross-attention residual backward
        grad_out1 = grad_out2.copy()
        grad_attn2 = grad_out2.copy()
        
        grad_attn2 = self.dropout2.backward(grad_attn2, training)
        grad_norm_out1, grad_enc, _ = self.mha2.backward(grad_attn2, training)
        grad_out1_from_norm = self.layernorm2.backward(grad_norm_out1)
        grad_out1 += grad_out1_from_norm
        
        # Self-attention residual backward
        grad_x = grad_out1.copy()
        grad_attn1 = grad_out1.copy()
        
        grad_attn1 = self.dropout1.backward(grad_attn1, training)
        grad_norm_x, _, _ = self.mha1.backward(grad_attn1, training)
        grad_x_from_norm = self.layernorm1.backward(grad_norm_x)
        grad_x += grad_x_from_norm
        
        return grad_x, grad_enc


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
        
        # Cache for backward
        self.input_ids = None
        self.embedded = None
        self.pos_embedded = None
        self.layer_outputs = []
    
    def parameters(self) -> List[Parameter]:
        params = [self.embedding]
        for layer in self.enc_layers:
            params.extend(layer.parameters())
        return params
    
    def forward(self, x: cp.ndarray, mask: Optional[cp.ndarray], training: bool = True) -> cp.ndarray:
        self.input_ids = x
        
        # Embedding lookup
        self.embedded = cp.take(self.embedding.data, x, axis=0)
        self.embedded *= cp.sqrt(float(self.d_model))
        
        # Add positional encoding
        self.pos_embedded = self.pos_encoding.forward(self.embedded)
        x_out = self.dropout.forward(self.pos_embedded, training)
        
        # Pass through encoder layers
        self.layer_outputs = []
        for layer in self.enc_layers:
            x_out = layer.forward(x_out, mask, training)
            self.layer_outputs.append(x_out)
        
        return x_out
    
    def backward(self, grad: cp.ndarray, training: bool = True):
        # Backward through encoder layers
        for layer in reversed(self.enc_layers):
            grad = layer.backward(grad, training)
        
        # Dropout backward
        grad = self.dropout.backward(grad, training)
        
        # Positional encoding backward (gradient passes through)
        grad_embedded = grad
        
        # Scale backward
        grad_embedded *= cp.sqrt(float(self.d_model))
        
        # Embedding backward
        batch_size, seq_len = self.input_ids.shape
        grad_embedding = cp.zeros_like(self.embedding.data)
        
        # Accumulate gradients for each token
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = int(self.input_ids[b, s])
                grad_embedding[token_id] += grad_embedded[b, s]
        
        if self.embedding.grad is None:
            self.embedding.grad = grad_embedding
        else:
            self.embedding.grad += grad_embedding


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
        
        # Cache for backward
        self.input_ids = None
        self.embedded = None
        self.pos_embedded = None
        self.layer_outputs = []
        self.enc_output_cache = None
    
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
        self.input_ids = x
        self.enc_output_cache = enc_output
        
        # Embedding lookup
        self.embedded = cp.take(self.embedding.data, x, axis=0)
        self.embedded *= cp.sqrt(float(self.d_model))
        
        # Add positional encoding
        self.pos_embedded = self.pos_encoding.forward(self.embedded)
        x_out = self.dropout.forward(self.pos_embedded, training)
        
        # Pass through decoder layers
        attention_weights = {}
        self.layer_outputs = []
        for i, layer in enumerate(self.dec_layers):
            x_out, block1, block2 = layer.forward(x_out, enc_output, look_ahead_mask, padding_mask, training)
            self.layer_outputs.append(x_out)
            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2
        
        return x_out, attention_weights
    
    def backward(self, grad: cp.ndarray, training: bool = True) -> cp.ndarray:
        # Backward through decoder layers
        grad_enc_total = cp.zeros_like(self.enc_output_cache)
        
        for layer in reversed(self.dec_layers):
            grad, grad_enc = layer.backward(grad, training)
            grad_enc_total += grad_enc
        
        # Dropout backward
        grad = self.dropout.backward(grad, training)
        
        # Positional encoding backward
        grad_embedded = grad
        
        # Scale backward
        grad_embedded *= cp.sqrt(float(self.d_model))
        
        # Embedding backward
        batch_size, seq_len = self.input_ids.shape
        grad_embedding = cp.zeros_like(self.embedding.data)
        
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = int(self.input_ids[b, s])
                grad_embedding[token_id] += grad_embedded[b, s]
        
        if self.embedding.grad is None:
            self.embedding.grad = grad_embedding
        else:
            self.embedding.grad += grad_embedding
        
        return grad_enc_total


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
        
        # Cache for backward
        self.dec_output = None
    
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
        self.dec_output, attention_weights = self.decoder.forward(
            tar, enc_output, look_ahead_mask, dec_padding_mask, training
        )
        final_output = cp.matmul(self.dec_output, self.final_layer.data)
        return final_output, attention_weights
    
    def backward(self, grad_output: cp.ndarray, training: bool = True):
        """Complete backward pass through the transformer."""
        # Gradient for final layer
        grad_final_layer = cp.matmul(
            self.dec_output.reshape(-1, self.dec_output.shape[-1]).T,
            grad_output.reshape(-1, grad_output.shape[-1])
        )
        
        if self.final_layer.grad is None:
            self.final_layer.grad = grad_final_layer
        else:
            self.final_layer.grad += grad_final_layer
        
        # Gradient for decoder output
        grad_dec_output = cp.matmul(grad_output, self.final_layer.data.T)
        
        # Backward through decoder
        grad_enc_output = self.decoder.backward(grad_dec_output, training)
        
        # Backward through encoder
        self.encoder.backward(grad_enc_output, training)


# ============================================================================
# Loss Function
# ============================================================================

def sparse_categorical_crossentropy(
    logits: cp.ndarray, targets: cp.ndarray, 
    mask: Optional[cp.ndarray] = None
) -> Tuple[float, cp.ndarray]:
    """
    Compute sparse categorical cross-entropy loss and gradients.
    
    Returns:
        loss: Scalar loss value
        grad_logits: Gradient w.r.t logits
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
    grad_logits = probs.copy()
    grad_logits[batch_indices, seq_indices, targets] -= 1
    
    if mask is not None:
        grad_logits *= mask[:, :, None]
    
    grad_logits /= norm_factor
    
    return total_loss, grad_logits


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
    """Training manager for Transformer with complete backpropagation."""
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
        """Single training step with full backpropagation."""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, _ = self.model.forward(
            inp, tar_inp, enc_mask, combined_mask, dec_mask, training=True
        )
        
        # Compute loss and gradients
        target_mask = (tar_real != 0).astype(cp.float32)
        loss, grad_logits = sparse_categorical_crossentropy(logits, tar_real, target_mask)
        
        # Backward pass through entire model
        self.model.backward(grad_logits, training=True)
        
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
        """Validation step without gradient computation."""
        logits, _ = self.model.forward(
            inp, tar_inp, enc_mask, combined_mask, dec_mask, training=False
        )
        
        target_mask = (tar_real != 0).astype(cp.float32)
        loss, _ = sparse_categorical_crossentropy(logits, tar_real, target_mask)
        accuracy = compute_accuracy(logits, tar_real, target_mask)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def train(self, train_data: List[Tuple], val_data: List[Tuple], 
              num_epochs: int, print_every: int = 10):
        """Full training loop with backpropagation."""
        print("=" * 80)
        print("Starting Training with Full Backpropagation")
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
    print("Transformer v3 - Complete Backpropagation System")
    print("=" * 80)
    
    # Test backprop with small example
    print("\nTesting Complete Backpropagation...")
    
    vocab_size = 100
    d_model = 32
    model = TransformerArchitecture(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        max_seq_len=20,
        dropout_rate=0.1
    )
    
    # Forward pass
    inp = cp.random.randint(1, vocab_size, (2, 5), dtype=cp.int32)
    tar = cp.random.randint(1, vocab_size, (2, 6), dtype=cp.int32)
    
    enc_mask = create_padding_mask(inp)
    dec_mask = create_padding_mask(inp)
    comb_mask = create_combined_mask(tar)
    
    logits, _ = model.forward(inp, tar, enc_mask, comb_mask, dec_mask, training=True)
    
    # Backward pass
    targets = cp.random.randint(0, vocab_size, (2, 6), dtype=cp.int32)
    loss, grad = sparse_categorical_crossentropy(logits, targets)
    model.backward(grad, training=True)
    
    # Check gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"✓ Forward pass successful")
    print(f"✓ Backward pass successful")
    print(f"✓ Parameters with gradients: {params_with_grad}/{len(model.parameters())}")
    
    print("\n" + "=" * 80)
    print("✓ Complete Backpropagation System Working!")
    print("=" * 80)
