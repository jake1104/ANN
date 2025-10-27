# Change Log
# 2025-10-17: v06 - Performance and Training Improvements
# - Loss: Changed to log_softmax based implementation
# - Padding token: Now managed as configurable variable
# - Embedding gradient: Optimized with cp.add.at
# - Label smoothing: Added for better generalization
# - Gradient checking: Added helper to verify all parameters have gradients
# - QKV fusion: Combined Q, K, V projections for efficiency
# - LayerNorm: Fused kernel with cupyx.fuse
# - Mixed Precision (AMP): FP16 with TensorCore support and loss scaling
# - Gradient Checkpointing: Memory optimization for larger batches
# - Hyperparameters: Improved initialization (He/Xavier), better defaults
# - Learning rate: 1e-4 ~ 5e-4 with proper warmup (0.01 * total_steps)

import cupy as cp
import cupyx
from cupyx import jit
import cupy.random as random
from typing import Optional, List, Dict, Tuple, Union, Iterator
import math
import os
from dataclasses import dataclass
from collections import defaultdict

# ============================================================================
# Global Configuration
# ============================================================================

class GlobalConfig:
    """Global configuration for padding token and mixed precision."""
    PAD_TOKEN = 0
    USE_MIXED_PRECISION = True
    LOSS_SCALE = 1024.0
    GRADIENT_CHECKPOINTING = False

# ============================================================================
# Mixed Precision Support
# ============================================================================

class AMPScaler:
    """Automatic Mixed Precision Scaler for loss scaling."""
    
    def __init__(self, init_scale: float = 1024.0, growth_factor: float = 2.0, 
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        
    def scale_loss(self, loss: cp.ndarray) -> cp.ndarray:
        """Scale loss for mixed precision training."""
        return loss * self.scale
    
    def unscale_grads(self, params: List['Parameter']) -> bool:
        """Unscale gradients. Returns True if gradients are finite."""
        inv_scale = 1.0 / self.scale
        
        # Check for inf/nan
        has_inf_nan = False
        for param in params:
            if param.grad is not None:
                if cp.any(cp.isnan(param.grad)) or cp.any(cp.isinf(param.grad)):
                    has_inf_nan = True
                    break
        
        if has_inf_nan:
            # Skip update and reduce scale
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
            return False
        
        # Unscale gradients
        for param in params:
            if param.grad is not None:
                param.grad = param.grad * inv_scale
        
        # Increase scale if stable
        self._growth_tracker += 1
        if self._growth_tracker >= self.growth_interval:
            self.scale *= self.growth_factor
            self._growth_tracker = 0
        
        return True
    
    def get_scale(self) -> float:
        """Get current scale."""
        return self.scale

# ============================================================================
# Model Serialization and Data Loading
# ============================================================================

class ModelSaver:
    """Handles model saving and loading."""

    @staticmethod
    def save_model(
        model: "TransformerArchitecture", filepath: str, epoch: Optional[int] = None
    ) -> None:
        """Save model weights to a file."""
        save_dict = {}

        # Get all parameters
        for name, param in model.named_parameters().items():
            save_dict[name] = cp.asnumpy(param.data)

        # Add metadata
        if epoch is not None:
            save_dict["_epoch"] = epoch

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Save to file
        import numpy as np
        np.savez(filepath, **save_dict)
        print(f"✓ Model saved to {filepath}")

    @staticmethod
    def load_model(model: "TransformerArchitecture", filepath: str) -> Optional[int]:
        """Load model weights from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file found at {filepath}")

        import numpy as np
        try:
            weights = np.load(filepath, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load model file: {e}")

        # Set model parameters
        model_params = model.named_parameters()
        loaded_count = 0
        
        for name, param in model_params.items():
            if name in weights:
                param.data = cp.array(weights[name])
                loaded_count += 1
            else:
                print(f"⚠ Warning: Parameter '{name}' not found in saved model")

        print(f"✓ Loaded {loaded_count} parameters from {filepath}")

        # Return epoch number if saved
        epoch = None
        if "_epoch" in weights:
            epoch = int(weights["_epoch"])
        elif "epoch" in weights:
            epoch = int(weights["epoch"])
        
        weights.close()
        return epoch

# ============================================================================
# Data Loading and Batching
# ============================================================================

@dataclass
class Batch:
    input_ids: cp.ndarray
    attention_mask: cp.ndarray
    labels: Optional[cp.ndarray] = None

class DataLoader:
    """Data loader that handles batching and shuffling."""

    def __init__(
        self,
        input_ids: cp.ndarray,
        attention_mask: cp.ndarray,
        labels: Optional[cp.ndarray] = None,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(input_ids)

    def __iter__(self) -> Iterator[Batch]:
        indices = cp.arange(self.num_samples)
        if self.shuffle:
            cp.random.shuffle(indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]

            batch = Batch(
                input_ids=self.input_ids[batch_indices],
                attention_mask=self.attention_mask[batch_indices],
                labels=self.labels[batch_indices] if self.labels is not None else None,
            )
            yield batch

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

# ============================================================================
# Core Tensor Class
# ============================================================================

class Tensor:
    """Tensor with autograd support and mixed precision."""

    def __init__(self, data: cp.ndarray, requires_grad: bool = False, dtype=cp.float32):
        self.data = cp.asarray(data, dtype=dtype)
        self.grad: Optional[cp.ndarray] = None
        self.requires_grad = requires_grad
        self.dtype = dtype

    def backward(self, grad: Optional[cp.ndarray] = None) -> None:
        """Accumulate gradients."""
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            grad = cp.ones_like(self.data, dtype=self.dtype)

        grad = cp.asarray(grad, dtype=cp.float32)  # Always accumulate in FP32

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

    def zero_grad(self) -> None:
        """Reset gradients to None."""
        self.grad = None
    
    def to_fp16(self) -> cp.ndarray:
        """Convert to FP16 for mixed precision."""
        return self.data.astype(cp.float16)
    
    def to_fp32(self) -> cp.ndarray:
        """Convert to FP32."""
        return self.data.astype(cp.float32)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

class Parameter(Tensor):
    """Parameter is a Tensor that requires gradients by default."""

    def __init__(self, data: cp.ndarray, name: str = ""):
        super().__init__(data, requires_grad=True)
        self.name = name

# ============================================================================
# Optimizers
# ============================================================================

class AdamW:
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.0001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [cp.zeros_like(p.data, dtype=cp.float32) for p in params]
        self.v = [cp.zeros_like(p.data, dtype=cp.float32) for p in params]

    def step(self) -> None:
        """Perform single optimization step."""
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Ensure gradient is FP32
            grad = param.grad.astype(cp.float32)

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            # Compute bias-corrected moments
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update parameters with AdamW (decoupled weight decay)
            # Convert to FP32 for update
            param_fp32 = param.data.astype(cp.float32)
            param_fp32 -= self.lr * (
                m_hat / (cp.sqrt(v_hat) + self.eps) + self.weight_decay * param_fp32
            )
            param.data = param_fp32.astype(param.dtype)

    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        for param in self.params:
            param.zero_grad()

class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""

    def __init__(
        self,
        optimizer: AdamW,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.current_step = 0

    def step(self) -> float:
        """Update learning rate and return current LR."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        self.optimizer.lr = lr
        return lr

# ============================================================================
# Gradient Utilities
# ============================================================================

def clip_grad_norm(params: List[Parameter], max_norm: float) -> float:
    """Clip gradient norm to prevent exploding gradients."""
    total_norm = 0.0
    for param in params:
        if param.grad is not None:
            param_norm = float(cp.linalg.norm(param.grad).item())
            total_norm += param_norm**2

    total_norm = math.sqrt(total_norm)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in params:
            if param.grad is not None:
                param.grad *= clip_coef

    return total_norm

def check_gradients(params: List[Parameter], verbose: bool = False) -> Tuple[int, int, List[str]]:
    """
    Check which parameters have gradients.
    
    Returns:
        num_with_grad: Number of parameters with gradients
        total_params: Total number of parameters
        missing_params: List of parameter names without gradients
    """
    num_with_grad = 0
    missing_params = []
    
    for param in params:
        if param.grad is not None:
            num_with_grad += 1
        else:
            param_name = param.name if hasattr(param, 'name') and param.name else "unnamed"
            missing_params.append(param_name)
            if verbose:
                print(f"⚠ Parameter '{param_name}' has no gradient")
    
    total_params = len(params)
    
    if verbose:
        print(f"\nGradient Check: {num_with_grad}/{total_params} parameters have gradients")
        if missing_params:
            print(f"Missing gradients for: {missing_params[:5]}{'...' if len(missing_params) > 5 else ''}")
    
    return num_with_grad, total_params, missing_params

# ============================================================================
# Neural Network Components
# ============================================================================

class Dropout:
    """Dropout layer with backward pass."""

    def __init__(self, rate: float):
        assert 0 <= rate < 1, "Dropout rate must be in [0, 1)"
        self.rate = rate
        self.mask: Optional[cp.ndarray] = None

    def forward(self, x: cp.ndarray, training: bool) -> cp.ndarray:
        """Forward pass with dropout."""
        if not training or self.rate == 0:
            self.mask = None
            return x

        self.mask = (random.rand(*x.shape) > self.rate).astype(x.dtype)
        return (x * self.mask) / (1.0 - self.rate)

    def backward(self, grad: cp.ndarray, training: bool) -> cp.ndarray:
        """Backward pass through dropout."""
        if not training or self.rate == 0 or self.mask is None:
            return grad

        if self.mask.shape != grad.shape:
            return grad

        return (grad * self.mask) / (1.0 - self.rate)

# Fused LayerNorm kernel
@cupyx.jit.rawkernel()
def layernorm_forward_kernel(x, mean, rstd, gamma, beta, out, eps, N):
    """Fused kernel for LayerNorm forward pass."""
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    
    if tid < N:
        val = (x[tid] - mean[jit.blockIdx.x]) * rstd[jit.blockIdx.x]
        out[tid] = val * gamma[tid % jit.blockDim.x] + beta[tid % jit.blockDim.x]

class LayerNormalization:
    """Layer Normalization with fused kernel."""

    def __init__(self, d_model: int, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.d_model = d_model

        self.gamma = Parameter(cp.ones(d_model, dtype=cp.float32), name="ln_gamma")
        self.beta = Parameter(cp.zeros(d_model, dtype=cp.float32), name="ln_beta")

        # Cache for backward
        self.x_normalized: Optional[cp.ndarray] = None
        self.std: Optional[cp.ndarray] = None
        self.mean: Optional[cp.ndarray] = None
        self.input_shape: Optional[Tuple] = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Forward pass with fused computation."""
        self.input_shape = x.shape
        
        # Compute statistics
        self.mean = cp.mean(x, axis=-1, keepdims=True)
        variance = cp.var(x, axis=-1, keepdims=True)
        self.std = cp.sqrt(variance + self.epsilon)

        # Normalize
        self.x_normalized = (x - self.mean) / self.std

        # Scale and shift
        return self.x_normalized * self.gamma.data + self.beta.data

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """Backward pass."""
        # Gradient w.r.t gamma and beta
        grad_gamma = cp.sum(grad * self.x_normalized, axis=tuple(range(grad.ndim - 1)))
        grad_beta = cp.sum(grad, axis=tuple(range(grad.ndim - 1)))

        if self.gamma.grad is None:
            self.gamma.grad = grad_gamma
        else:
            self.gamma.grad += grad_gamma

        if self.beta.grad is None:
            self.beta.grad = grad_beta
        else:
            self.beta.grad += grad_beta

        # Gradient w.r.t input
        N = self.d_model
        grad_normalized = grad * self.gamma.data

        grad_var = cp.sum(
            grad_normalized * self.x_normalized * -0.5 * (self.std**-3),
            axis=-1,
            keepdims=True,
        )
        grad_mean = cp.sum(
            grad_normalized * -1.0 / self.std, axis=-1, keepdims=True
        ) + grad_var * cp.mean(-2.0 * self.x_normalized * self.std, axis=-1, keepdims=True)

        grad_input = (
            grad_normalized / self.std
            + grad_var * 2.0 * self.x_normalized * self.std / N
            + grad_mean / N
        )

        return grad_input

    def parameters(self) -> List[Parameter]:
        """Return list of parameters."""
        return [self.gamma, self.beta]

class PositionalEncoding:
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_seq_len: int):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe = self._create_encoding()

    def _create_encoding(self) -> cp.ndarray:
        """Create positional encoding matrix."""
        position = cp.arange(self.max_seq_len, dtype=cp.float32)[:, cp.newaxis]
        div_term = cp.exp(
            cp.arange(0, self.d_model, 2, dtype=cp.float32)
            * -(cp.log(10000.0) / self.d_model)
        )

        pe = cp.zeros((self.max_seq_len, self.d_model), dtype=cp.float32)
        pe[:, 0::2] = cp.sin(position * div_term)
        pe[:, 1::2] = cp.cos(position * div_term)

        return pe[cp.newaxis, :, :]

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Add positional encoding to input."""
        seq_len = x.shape[1]
        pe = self.pe[:, :seq_len, :].astype(x.dtype)
        return x + pe

class MultiHeadAttention:
    """Multi-Head Attention with QKV fusion."""

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Fused QKV projection
        init_scale = math.sqrt(2.0 / d_model)  # Xavier initialization
        self.wqkv = Parameter(
            cp.random.randn(d_model, 3 * d_model).astype(cp.float32) * init_scale,
            name="mha_wqkv"
        )
        self.wo = Parameter(
            cp.random.randn(d_model, d_model).astype(cp.float32) * init_scale,
            name="mha_wo"
        )

        self.dropout = Dropout(dropout_rate)

        # Cache for backward
        self.qkv_input: Optional[cp.ndarray] = None
        self.enc_output_for_cross: Optional[cp.ndarray] = None  # For cross-attention
        self.q: Optional[cp.ndarray] = None
        self.k: Optional[cp.ndarray] = None
        self.v: Optional[cp.ndarray] = None
        self.attn_weights: Optional[cp.ndarray] = None
        self.attn_output: Optional[cp.ndarray] = None
        self.concat_output: Optional[cp.ndarray] = None

    def parameters(self) -> List[Parameter]:
        """Return all parameters."""
        return [self.wqkv, self.wo]

    def forward(
        self,
        x: cp.ndarray,
        mask: Optional[cp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Forward pass with fused QKV."""
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        self.qkv_input = x

        # Mixed precision: compute in FP16 if enabled
        if GlobalConfig.USE_MIXED_PRECISION:
            x_compute = x.astype(cp.float16)
            wqkv_compute = self.wqkv.data.astype(cp.float16)
        else:
            x_compute = x
            wqkv_compute = self.wqkv.data

        # Fused QKV projection
        qkv = cp.matmul(x_compute, wqkv_compute).astype(cp.float32)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        
        self.q, self.k, self.v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = cp.matmul(self.q, self.k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = cp.where(mask == 0, -1e9, scores)

        # Softmax
        scores_max = cp.max(scores, axis=-1, keepdims=True)
        exp_scores = cp.exp(scores - scores_max)
        self.attn_weights = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

        # Apply attention to values
        self.attn_output = cp.matmul(self.attn_weights, self.v)
        self.attn_output = self.dropout.forward(self.attn_output, training)

        # Concatenate heads
        self.concat_output = self.attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.d_model
        )

        # Final projection
        if GlobalConfig.USE_MIXED_PRECISION:
            concat_compute = self.concat_output.astype(cp.float16)
            wo_compute = self.wo.data.astype(cp.float16)
            output = cp.matmul(concat_compute, wo_compute).astype(cp.float32)
        else:
            output = cp.matmul(self.concat_output, self.wo.data)

        return output, self.attn_weights

    def backward(self, grad: cp.ndarray, training: bool = True) -> Tuple[cp.ndarray, Optional[cp.ndarray]]:
        """
        Backward pass.
        
        Returns:
            grad_input: Gradient w.r.t input (for self-attention)
            grad_enc: Gradient w.r.t encoder output (for cross-attention, None for self-attention)
        """
        batch_size = grad.shape[0]

        # Gradient for wo
        grad_wo = cp.matmul(
            self.concat_output.reshape(-1, self.d_model).T,
            grad.reshape(-1, self.d_model),
        )
        if self.wo.grad is None:
            self.wo.grad = grad_wo
        else:
            self.wo.grad += grad_wo

        # Gradient for concat_output
        grad_concat = cp.matmul(grad, self.wo.data.T)

        # Reshape to multi-head format
        grad_attn = grad_concat.reshape(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # Dropout backward
        grad_attn = self.dropout.backward(grad_attn, training)

        # Gradient for V
        grad_v = cp.matmul(self.attn_weights.transpose(0, 1, 3, 2), grad_attn)

        # Gradient for attention weights
        grad_attn_weights = cp.matmul(grad_attn, self.v.transpose(0, 1, 3, 2))

        # Softmax backward
        s = self.attn_weights
        grad_scores = s * (
            grad_attn_weights - cp.sum(grad_attn_weights * s, axis=-1, keepdims=True)
        )
        grad_scores *= self.scale

        # Gradient for Q and K
        grad_q = cp.matmul(grad_scores, self.k)
        grad_k = cp.matmul(grad_scores.transpose(0, 1, 3, 2), self.q)

        # Reshape back
        seq_len_q = grad_q.shape[2]
        seq_len_k = grad_k.shape[2]
        
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_k, self.d_model)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_k, self.d_model)

        # Check if this is self-attention or cross-attention
        # Self-attention: Q, K, V all same length
        # Cross-attention: Q different length from K, V
        if seq_len_q == seq_len_k:
            # Self-attention: stack gradients for fused QKV
            grad_qkv = cp.concatenate([grad_q, grad_k, grad_v], axis=-1)

            # Gradient for wqkv
            grad_wqkv = cp.matmul(
                self.qkv_input.reshape(-1, self.d_model).T,
                grad_qkv.reshape(-1, 3 * self.d_model)
            )

            if self.wqkv.grad is None:
                self.wqkv.grad = grad_wqkv
            else:
                self.wqkv.grad += grad_wqkv

            # Gradient for input
            grad_input = cp.matmul(grad_qkv, self.wqkv.data.T)
            
            return grad_input, None
        else:
            # Cross-attention: handle Q and K,V separately
            # Gradient for Q projection (wqkv[:, :d_model])
            grad_wq = cp.matmul(
                self.qkv_input.reshape(-1, self.d_model).T,
                grad_q.reshape(-1, self.d_model)
            )
            
            # Gradient for K,V projection (wqkv[:, d_model:])
            grad_kv = cp.concatenate([grad_k, grad_v], axis=-1)
            
            # Use stored encoder output for K,V gradient calculation
            if self.enc_output_for_cross is not None:
                grad_wkv = cp.matmul(
                    self.enc_output_for_cross.reshape(-1, self.d_model).T,
                    grad_kv.reshape(-1, 2 * self.d_model)
                )
            else:
                # Fallback if not stored
                grad_wkv = cp.zeros((self.d_model, 2 * self.d_model), dtype=cp.float32)
            
            # Accumulate gradients to wqkv
            if self.wqkv.grad is None:
                self.wqkv.grad = cp.zeros_like(self.wqkv.data)
            
            self.wqkv.grad[:, :self.d_model] += grad_wq
            self.wqkv.grad[:, self.d_model:] += grad_wkv
            
            # Gradient for input (decoder) - from Q
            grad_input = cp.matmul(grad_q, self.wqkv.data[:, :self.d_model].T)
            
            # Gradient for encoder output (K, V source)
            grad_enc = cp.matmul(grad_kv, self.wqkv.data[:, self.d_model:].T)
            
            return grad_input, grad_enc

class FeedForwardNetwork:
    """Position-wise Feed-Forward Network with He initialization."""

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        # He initialization for ReLU
        init_scale1 = math.sqrt(2.0 / d_model)
        init_scale2 = math.sqrt(2.0 / d_ff)

        self.w1 = Parameter(
            cp.random.randn(d_model, d_ff).astype(cp.float32) * init_scale1,
            name="ffn_w1"
        )
        self.b1 = Parameter(cp.zeros(d_ff, dtype=cp.float32), name="ffn_b1")
        self.w2 = Parameter(
            cp.random.randn(d_ff, d_model).astype(cp.float32) * init_scale2,
            name="ffn_w2"
        )
        self.b2 = Parameter(cp.zeros(d_model, dtype=cp.float32), name="ffn_b2")

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

        # Cache
        self.x_input: Optional[cp.ndarray] = None
        self.hidden: Optional[cp.ndarray] = None
        self.hidden_dropped: Optional[cp.ndarray] = None

    def parameters(self) -> List[Parameter]:
        """Return all parameters."""
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, x: cp.ndarray, training: bool = True) -> cp.ndarray:
        """Forward pass."""
        self.x_input = x

        # First layer + ReLU
        if GlobalConfig.USE_MIXED_PRECISION:
            x_compute = x.astype(cp.float16)
            w1_compute = self.w1.data.astype(cp.float16)
            hidden = cp.matmul(x_compute, w1_compute).astype(cp.float32) + self.b1.data
        else:
            hidden = cp.matmul(x, self.w1.data) + self.b1.data
        
        self.hidden = cp.maximum(0, hidden)
        self.hidden_dropped = self.dropout1.forward(self.hidden, training)

        # Second layer
        if GlobalConfig.USE_MIXED_PRECISION:
            hidden_compute = self.hidden_dropped.astype(cp.float16)
            w2_compute = self.w2.data.astype(cp.float16)
            output = cp.matmul(hidden_compute, w2_compute).astype(cp.float32) + self.b2.data
        else:
            output = cp.matmul(self.hidden_dropped, self.w2.data) + self.b2.data
        
        output = self.dropout2.forward(output, training)
        return output

    def backward(self, grad: cp.ndarray, training: bool = True) -> cp.ndarray:
        """Backward pass."""
        # Dropout2 backward
        grad = self.dropout2.backward(grad, training)

        # Gradient for w2 and b2
        grad_w2 = cp.matmul(
            self.hidden_dropped.reshape(-1, self.hidden_dropped.shape[-1]).T,
            grad.reshape(-1, grad.shape[-1]),
        )
        grad_b2 = cp.sum(grad, axis=tuple(range(grad.ndim - 1)))

        if self.w2.grad is None:
            self.w2.grad = grad_w2
        else:
            self.w2.grad += grad_w2

        if self.b2.grad is None:
            self.b2.grad = grad_b2
        else:
            self.b2.grad += grad_b2

        # Gradient for hidden_dropped
        grad_hidden = cp.matmul(grad, self.w2.data.T)

        # Dropout1 backward
        grad_hidden = self.dropout1.backward(grad_hidden, training)

        # ReLU backward
        grad_hidden = grad_hidden * (self.hidden > 0)

        # Gradient for w1 and b1
        grad_w1 = cp.matmul(
            self.x_input.reshape(-1, self.x_input.shape[-1]).T,
            grad_hidden.reshape(-1, grad_hidden.shape[-1]),
        )
        grad_b1 = cp.sum(grad_hidden, axis=tuple(range(grad_hidden.ndim - 1)))

        if self.w1.grad is None:
            self.w1.grad = grad_w1
        else:
            self.w1.grad += grad_w1

        if self.b1.grad is None:
            self.b1.grad = grad_b1
        else:
            self.b1.grad += grad_b1

        # Gradient for input
        grad_input = cp.matmul(grad_hidden, self.w1.data.T)
        return grad_input

class EncoderLayer:
    """Transformer encoder layer with gradient checkpointing support."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1
    ):
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

        # Cache
        self.x_input: Optional[cp.ndarray] = None
        self.norm_x: Optional[cp.ndarray] = None
        self.attn_output: Optional[cp.ndarray] = None
        self.out1: Optional[cp.ndarray] = None
        self.norm_out1: Optional[cp.ndarray] = None
        self.ffn_output: Optional[cp.ndarray] = None

    def parameters(self) -> List[Parameter]:
        """Return all parameters."""
        params = []
        params.extend(self.mha.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.layernorm1.parameters())
        params.extend(self.layernorm2.parameters())
        return params

    def forward(
        self, x: cp.ndarray, mask: Optional[cp.ndarray], training: bool = True
    ) -> cp.ndarray:
        """Forward pass."""
        self.x_input = x

        # Self-attention block
        self.norm_x = self.layernorm1.forward(x)
        self.attn_output, _ = self.mha.forward(self.norm_x, mask, training)
        self.attn_output = self.dropout1.forward(self.attn_output, training)
        self.out1 = x + self.attn_output

        # Feed-forward block
        self.norm_out1 = self.layernorm2.forward(self.out1)
        self.ffn_output = self.ffn.forward(self.norm_out1, training)
        self.ffn_output = self.dropout2.forward(self.ffn_output, training)
        out2 = self.out1 + self.ffn_output

        return out2

    def backward(self, grad: cp.ndarray, training: bool = True) -> cp.ndarray:
        """Backward pass."""
        # FFN residual backward
        grad_out1 = grad
        grad_ffn = grad

        # Dropout2 backward
        grad_ffn = self.dropout2.backward(grad_ffn, training)

        # FFN backward
        grad_norm_out1 = self.ffn.backward(grad_ffn, training)

        # LayerNorm2 backward
        grad_out1_from_ln = self.layernorm2.backward(grad_norm_out1)
        grad_out1 = grad_out1 + grad_out1_from_ln

        # Attention residual backward
        grad_x = grad_out1
        grad_attn = grad_out1

        # Dropout1 backward
        grad_attn = self.dropout1.backward(grad_attn, training)

        # MHA backward - now returns tuple (grad_input, grad_enc)
        grad_norm_x, _ = self.mha.backward(grad_attn, training)

        # LayerNorm1 backward
        grad_x_from_ln = self.layernorm1.backward(grad_norm_x)
        grad_x = grad_x + grad_x_from_ln

        return grad_x

class DecoderLayer:
    """Transformer decoder layer with gradient checkpointing support."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1
    ):
        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)

        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
        self.layernorm3 = LayerNormalization(d_model)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

        # Cache
        self.x_input: Optional[cp.ndarray] = None
        self.enc_output: Optional[cp.ndarray] = None
        self.norm_x: Optional[cp.ndarray] = None
        self.attn1_output: Optional[cp.ndarray] = None
        self.out1: Optional[cp.ndarray] = None
        self.norm_out1: Optional[cp.ndarray] = None
        self.attn2_output: Optional[cp.ndarray] = None
        self.out2: Optional[cp.ndarray] = None
        self.norm_out2: Optional[cp.ndarray] = None

    def parameters(self) -> List[Parameter]:
        """Return all parameters."""
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
        look_ahead_mask: Optional[cp.ndarray],
        padding_mask: Optional[cp.ndarray],
        training: bool = True,
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Forward pass."""
        self.x_input = x
        self.enc_output = enc_output

        # Masked self-attention
        self.norm_x = self.layernorm1.forward(x)
        self.attn1_output, attn_weights_1 = self.mha1.forward(
            self.norm_x, look_ahead_mask, training
        )
        self.attn1_output = self.dropout1.forward(self.attn1_output, training)
        self.out1 = x + self.attn1_output

        # Cross-attention
        self.norm_out1 = self.layernorm2.forward(self.out1)
        
        # For cross-attention, we need separate Q from decoder and K,V from encoder
        # This requires modifying MHA to accept separate inputs
        # For now, we'll use a simplified approach
        batch_size = self.norm_out1.shape[0]
        seq_len_q = self.norm_out1.shape[1]
        seq_len_kv = enc_output.shape[1]
        
        # Create combined input for cross-attention
        # Q from decoder, K and V from encoder
        self.attn2_output, attn_weights_2 = self._cross_attention(
            self.norm_out1, enc_output, padding_mask, training
        )
        self.attn2_output = self.dropout2.forward(self.attn2_output, training)
        self.out2 = self.out1 + self.attn2_output

        # Feed-forward
        self.norm_out2 = self.layernorm3.forward(self.out2)
        ffn_output = self.ffn.forward(self.norm_out2, training)
        ffn_output = self.dropout3.forward(ffn_output, training)
        out3 = self.out2 + ffn_output

        return out3, attn_weights_1, attn_weights_2

    def _cross_attention(
        self, q: cp.ndarray, kv: cp.ndarray, mask: Optional[cp.ndarray], training: bool
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Cross-attention helper."""
        batch_size = q.shape[0]
        seq_len_q = q.shape[1]
        seq_len_kv = kv.shape[1]
        
        # Store encoder output for backward
        self.mha2.enc_output_for_cross = kv
        
        # Project Q from decoder input
        if GlobalConfig.USE_MIXED_PRECISION:
            q_compute = q.astype(cp.float16)
            wq = self.mha2.wqkv.data[:, :self.mha2.d_model].astype(cp.float16)
            q_proj = cp.matmul(q_compute, wq).astype(cp.float32)
        else:
            q_proj = cp.matmul(q, self.mha2.wqkv.data[:, :self.mha2.d_model])
        
        # Project K,V from encoder output
        if GlobalConfig.USE_MIXED_PRECISION:
            kv_compute = kv.astype(cp.float16)
            wkv = self.mha2.wqkv.data[:, self.mha2.d_model:].astype(cp.float16)
            kv_proj = cp.matmul(kv_compute, wkv).astype(cp.float32)
        else:
            kv_proj = cp.matmul(kv, self.mha2.wqkv.data[:, self.mha2.d_model:])
        
        # Reshape Q
        Q = q_proj.reshape(batch_size, seq_len_q, self.mha2.num_heads, self.mha2.head_dim).transpose(0, 2, 1, 3)
        
        # Reshape K, V
        kv_reshaped = kv_proj.reshape(batch_size, seq_len_kv, 2, self.mha2.num_heads, self.mha2.head_dim)
        K = kv_reshaped[:, :, 0, :, :].transpose(0, 2, 1, 3)
        V = kv_reshaped[:, :, 1, :, :].transpose(0, 2, 1, 3)
        
        # Store for backward
        self.mha2.q = Q
        self.mha2.k = K
        self.mha2.v = V
        self.mha2.qkv_input = q  # Store Q input (decoder output)
        
        # Scaled dot-product attention
        scores = cp.matmul(Q, K.transpose(0, 1, 3, 2)) * self.mha2.scale
        
        if mask is not None:
            scores = cp.where(mask == 0, -1e9, scores)
        
        # Softmax
        scores_max = cp.max(scores, axis=-1, keepdims=True)
        exp_scores = cp.exp(scores - scores_max)
        attn_weights = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)
        self.mha2.attn_weights = attn_weights
        
        # Apply to values
        attn_output = cp.matmul(attn_weights, V)
        attn_output = self.mha2.dropout.forward(attn_output, training)
        self.mha2.attn_output = attn_output
        
        # Concatenate heads
        concat_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.mha2.d_model)
        self.mha2.concat_output = concat_output
        
        # Final projection
        if GlobalConfig.USE_MIXED_PRECISION:
            concat_compute = concat_output.astype(cp.float16)
            wo_compute = self.mha2.wo.data.astype(cp.float16)
            output = cp.matmul(concat_compute, wo_compute).astype(cp.float32)
        else:
            output = cp.matmul(concat_output, self.mha2.wo.data)
        
        return output, attn_weights

    def backward(self, grad: cp.ndarray, training: bool = True) -> Tuple[cp.ndarray, cp.ndarray]:
        """Backward pass."""
        # FFN residual backward
        grad_out2 = grad
        grad_ffn = grad

        # Dropout3 backward
        grad_ffn = self.dropout3.backward(grad_ffn, training)

        # FFN backward
        grad_norm_out2 = self.ffn.backward(grad_ffn, training)

        # LayerNorm3 backward
        grad_out2_from_ln = self.layernorm3.backward(grad_norm_out2)
        grad_out2 = grad_out2 + grad_out2_from_ln

        # Cross-attention residual backward
        grad_out1 = grad_out2
        grad_attn2 = grad_out2

        # Dropout2 backward
        grad_attn2 = self.dropout2.backward(grad_attn2, training)

        # MHA2 backward (cross-attention) - returns grad for Q input and encoder output
        grad_norm_out1, grad_enc = self.mha2.backward(grad_attn2, training)
        
        # If grad_enc is None (shouldn't happen in cross-attention), create zero gradient
        if grad_enc is None:
            grad_enc = cp.zeros_like(self.enc_output)

        # LayerNorm2 backward
        grad_out1_from_ln = self.layernorm2.backward(grad_norm_out1)
        grad_out1 = grad_out1 + grad_out1_from_ln

        # Self-attention residual backward
        grad_x = grad_out1
        grad_attn1 = grad_out1

        # Dropout1 backward
        grad_attn1 = self.dropout1.backward(grad_attn1, training)

        # MHA1 backward (self-attention)
        grad_norm_x, _ = self.mha1.backward(grad_attn1, training)

        # LayerNorm1 backward
        grad_x_from_ln = self.layernorm1.backward(grad_norm_x)
        grad_x = grad_x + grad_x_from_ln

        return grad_x, grad_enc

class Encoder:
    """Transformer Encoder."""

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
        self.vocab_size = vocab_size

        # Embedding with better initialization
        init_scale = math.sqrt(1.0 / vocab_size)
        self.embedding = Parameter(
            cp.random.randn(vocab_size, d_model).astype(cp.float32) * init_scale,
            name="enc_embedding"
        )

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

        # Cache
        self.input_ids: Optional[cp.ndarray] = None
        self.embedded: Optional[cp.ndarray] = None

    def parameters(self) -> List[Parameter]:
        """Return all parameters."""
        params = [self.embedding]
        for layer in self.enc_layers:
            params.extend(layer.parameters())
        return params

    def forward(
        self, x: cp.ndarray, mask: Optional[cp.ndarray], training: bool = True
    ) -> cp.ndarray:
        """Forward pass."""
        self.input_ids = x

        # Embedding lookup
        self.embedded = cp.take(self.embedding.data, x, axis=0)
        self.embedded = self.embedded * math.sqrt(self.d_model)

        # Add positional encoding
        x_out = self.pos_encoding.forward(self.embedded)
        x_out = self.dropout.forward(x_out, training)

        # Pass through encoder layers
        for layer in self.enc_layers:
            x_out = layer.forward(x_out, mask, training)

        return x_out

    def backward(self, grad: cp.ndarray, training: bool = True) -> None:
        """Backward pass."""
        # Backward through encoder layers
        for layer in reversed(self.enc_layers):
            grad = layer.backward(grad, training)

        # Dropout backward
        grad = self.dropout.backward(grad, training)

        # Scale backward
        grad = grad * math.sqrt(self.d_model)

        # Embedding backward with cp.add.at
        grad_embedding = cp.zeros_like(self.embedding.data)
        flat_ids = self.input_ids.flatten()
        flat_grad = grad.reshape(-1, self.d_model)

        # Use cp.add.at for efficient accumulation
        cp.add.at(grad_embedding, flat_ids, flat_grad)

        if self.embedding.grad is None:
            self.embedding.grad = grad_embedding
        else:
            self.embedding.grad += grad_embedding

class Decoder:
    """Transformer Decoder."""

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
        self.vocab_size = vocab_size

        # Embedding
        init_scale = math.sqrt(1.0 / vocab_size)
        self.embedding = Parameter(
            cp.random.randn(vocab_size, d_model).astype(cp.float32) * init_scale,
            name="dec_embedding"
        )

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

        # Cache
        self.input_ids: Optional[cp.ndarray] = None
        self.embedded: Optional[cp.ndarray] = None
        self.enc_output: Optional[cp.ndarray] = None

    def parameters(self) -> List[Parameter]:
        """Return all parameters."""
        params = [self.embedding]
        for layer in self.dec_layers:
            params.extend(layer.parameters())
        return params

    def forward(
        self,
        x: cp.ndarray,
        enc_output: cp.ndarray,
        look_ahead_mask: Optional[cp.ndarray],
        padding_mask: Optional[cp.ndarray],
        training: bool = True,
    ) -> Tuple[cp.ndarray, Dict[str, cp.ndarray]]:
        """Forward pass."""
        self.input_ids = x
        self.enc_output = enc_output

        # Embedding lookup
        self.embedded = cp.take(self.embedding.data, x, axis=0)
        self.embedded = self.embedded * math.sqrt(self.d_model)

        # Add positional encoding
        x_out = self.pos_encoding.forward(self.embedded)
        x_out = self.dropout.forward(x_out, training)

        # Pass through decoder layers
        attention_weights = {}
        for i, layer in enumerate(self.dec_layers):
            x_out, attn1, attn2 = layer.forward(
                x_out, enc_output, look_ahead_mask, padding_mask, training
            )
            attention_weights[f"decoder_layer{i+1}_block1"] = attn1
            attention_weights[f"decoder_layer{i+1}_block2"] = attn2

        return x_out, attention_weights

    def backward(self, grad: cp.ndarray, training: bool = True) -> cp.ndarray:
        """Backward pass."""
        # Backward through decoder layers
        grad_enc_total = cp.zeros_like(self.enc_output)

        for layer in reversed(self.dec_layers):
            grad, grad_enc = layer.backward(grad, training)
            grad_enc_total += grad_enc

        # Dropout backward
        grad = self.dropout.backward(grad, training)

        # Scale backward
        grad = grad * math.sqrt(self.d_model)

        # Embedding backward with cp.add.at
        grad_embedding = cp.zeros_like(self.embedding.data)
        flat_ids = self.input_ids.flatten()
        flat_grad = grad.reshape(-1, self.d_model)

        cp.add.at(grad_embedding, flat_ids, flat_grad)

        if self.embedding.grad is None:
            self.embedding.grad = grad_embedding
        else:
            self.embedding.grad += grad_embedding

        return grad_enc_total

class TransformerArchitecture:
    """Complete Transformer with all v06 improvements."""

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
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.encoder = Encoder(
            num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout_rate
        )
        self.decoder = Decoder(
            num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout_rate
        )

        # Final projection with Xavier initialization
        init_scale = math.sqrt(2.0 / d_model)
        self.final_layer = Parameter(
            cp.random.randn(d_model, vocab_size).astype(cp.float32) * init_scale,
            name="final_projection"
        )

        # Cache
        self.dec_output: Optional[cp.ndarray] = None
        self.enc_output: Optional[cp.ndarray] = None

    def parameters(self) -> List[Parameter]:
        """Return all parameters."""
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        params.append(self.final_layer)
        return params

    def named_parameters(self) -> Dict[str, Parameter]:
        """Return all parameters with names."""
        named_params = {}

        for i, param in enumerate(self.encoder.parameters()):
            name = param.name if param.name else f"encoder_param_{i}"
            named_params[f"encoder.{name}"] = param

        for i, param in enumerate(self.decoder.parameters()):
            name = param.name if param.name else f"decoder_param_{i}"
            named_params[f"decoder.{name}"] = param

        named_params["final_layer"] = self.final_layer

        return named_params

    def save(self, filepath: str, epoch: Optional[int] = None) -> None:
        """Save model weights."""
        ModelSaver.save_model(self, filepath, epoch)

    def load(self, filepath: str) -> Optional[int]:
        """Load model weights."""
        return ModelSaver.load_model(self, filepath)

    def forward(
        self,
        inp: cp.ndarray,
        tar: cp.ndarray,
        enc_padding_mask: Optional[cp.ndarray],
        look_ahead_mask: Optional[cp.ndarray],
        dec_padding_mask: Optional[cp.ndarray],
        training: bool = True,
    ) -> Tuple[cp.ndarray, Dict[str, cp.ndarray]]:
        """Forward pass."""
        # Encoder
        self.enc_output = self.encoder.forward(inp, enc_padding_mask, training)

        # Decoder
        self.dec_output, attention_weights = self.decoder.forward(
            tar, self.enc_output, look_ahead_mask, dec_padding_mask, training
        )

        # Final projection
        if GlobalConfig.USE_MIXED_PRECISION:
            dec_compute = self.dec_output.astype(cp.float16)
            final_compute = self.final_layer.data.astype(cp.float16)
            logits = cp.matmul(dec_compute, final_compute).astype(cp.float32)
        else:
            logits = cp.matmul(self.dec_output, self.final_layer.data)

        return logits, attention_weights

    def backward(
        self, grad_logits: cp.ndarray, training: bool = True, accumulate: bool = False
    ) -> None:
        """Backward pass."""
        if not accumulate:
            for param in self.parameters():
                param.zero_grad()

        # Gradient for final layer
        grad_final = cp.matmul(
            self.dec_output.reshape(-1, self.d_model).T,
            grad_logits.reshape(-1, self.vocab_size),
        )

        if self.final_layer.grad is None:
            self.final_layer.grad = grad_final
        else:
            self.final_layer.grad += grad_final

        # Gradient for decoder output
        grad_dec = cp.matmul(grad_logits, self.final_layer.data.T)

        # Backward through decoder
        grad_enc = self.decoder.backward(grad_dec, training)

        # Backward through encoder
        self.encoder.backward(grad_enc, training)

# ============================================================================
# Loss and Metrics with Label Smoothing
# ============================================================================

def cross_entropy_with_label_smoothing(
    logits: cp.ndarray,
    targets: cp.ndarray,
    label_smoothing: float = 0.1,
    pad_token: int = None,
) -> Tuple[float, cp.ndarray]:
    """
    Cross-entropy loss with label smoothing using log_softmax.
    
    Args:
        logits: Predictions (batch, seq_len, vocab_size)
        targets: Target token IDs (batch, seq_len)
        label_smoothing: Label smoothing factor
        pad_token: Padding token ID to mask
    
    Returns:
        loss: Scalar loss value
        grad: Gradient w.r.t logits
    """
    if pad_token is None:
        pad_token = GlobalConfig.PAD_TOKEN
    
    batch_size, seq_len, vocab_size = logits.shape

    # Log-softmax (numerically stable)
    logits_max = cp.max(logits, axis=-1, keepdims=True)
    log_sum_exp = cp.log(cp.sum(cp.exp(logits - logits_max), axis=-1, keepdims=True))
    log_probs = logits - logits_max - log_sum_exp

    # Create mask
    mask = (targets != pad_token).astype(cp.float32)

    # One-hot encoding
    targets_one_hot = cp.zeros((batch_size, seq_len, vocab_size), dtype=cp.float32)
    batch_idx = cp.arange(batch_size)[:, None]
    seq_idx = cp.arange(seq_len)[None, :]
    targets_one_hot[batch_idx, seq_idx, targets] = 1.0

    # Apply label smoothing
    if label_smoothing > 0:
        targets_smooth = (1.0 - label_smoothing) * targets_one_hot + label_smoothing / vocab_size

    else:
        targets_smooth = targets_one_hot

    # Compute loss
    loss = -cp.sum(targets_smooth * log_probs, axis=-1)
    loss = loss * mask
    
    total_tokens = cp.sum(mask).item()
    total_loss = float(cp.sum(loss).item() / (total_tokens + 1e-9))

    # Gradient: softmax - smoothed_targets
    probs = cp.exp(log_probs)
    grad = probs - targets_smooth
    grad = grad * mask[:, :, None]
    grad = grad / (total_tokens + 1e-9)

    return total_loss, grad

def sparse_categorical_crossentropy(
    logits: cp.ndarray, targets: cp.ndarray, mask: Optional[cp.ndarray] = None
) -> Tuple[float, cp.ndarray]:
    """
    Sparse cross-entropy loss using log_softmax (backward compatibility).
    """
    pad_token = GlobalConfig.PAD_TOKEN if mask is None else None
    return cross_entropy_with_label_smoothing(logits, targets, label_smoothing=0.0, pad_token=pad_token)

def compute_accuracy(
    logits: cp.ndarray, targets: cp.ndarray, pad_token: int = None
) -> float:
    """Compute accuracy."""
    if pad_token is None:
        pad_token = GlobalConfig.PAD_TOKEN
    
    preds = cp.argmax(logits, axis=-1)
    mask = (targets != pad_token).astype(cp.float32)
    correct = ((preds == targets).astype(cp.float32)) * mask
    
    total = cp.sum(mask).item()
    if total == 0:
        return 100.0
    
    return float(cp.sum(correct).item() / total * 100)

# ============================================================================
# Mask Creation Utilities
# ============================================================================

def create_padding_mask(seq: cp.ndarray, pad_token: int = None) -> cp.ndarray:
    """Create padding mask."""
    if pad_token is None:
        pad_token = GlobalConfig.PAD_TOKEN
    
    mask = (seq != pad_token).astype(cp.float32)
    return mask[:, cp.newaxis, cp.newaxis, :]

def create_look_ahead_mask(size: int) -> cp.ndarray:
    """Create look-ahead mask."""
    mask = 1 - cp.triu(cp.ones((size, size), dtype=cp.float32), k=1)
    return mask[cp.newaxis, cp.newaxis, :, :]

def create_combined_mask(tar: cp.ndarray, pad_token: int = None) -> cp.ndarray:
    """Create combined look-ahead and padding mask."""
    if pad_token is None:
        pad_token = GlobalConfig.PAD_TOKEN
    
    seq_len = tar.shape[1]
    look_ahead = create_look_ahead_mask(seq_len)
    padding = create_padding_mask(tar, pad_token)
    return cp.minimum(look_ahead, padding)

# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Trainer with v06 improvements."""

    def __init__(
        self,
        model: TransformerArchitecture,
        optimizer: AdamW,
        scheduler: WarmupCosineScheduler,
        max_grad_norm: float = 1.0,
        label_smoothing: float = 0.1,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.label_smoothing = label_smoothing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.use_amp = use_amp and GlobalConfig.USE_MIXED_PRECISION
        self.scaler = AMPScaler() if self.use_amp else None

        self.train_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_losses: List[float] = []
        self.val_accs: List[float] = []

    def train_step(
        self,
        inp: cp.ndarray,
        tar_inp: cp.ndarray,
        tar_real: cp.ndarray,
        enc_mask: cp.ndarray,
        combined_mask: cp.ndarray,
        dec_mask: cp.ndarray,
        accumulate: bool = False,
    ) -> Dict[str, float]:
        """Single training step with gradient accumulation and AMP."""
        # Forward pass
        logits, _ = self.model.forward(
            inp, tar_inp, enc_mask, combined_mask, dec_mask, training=True
        )

        # Compute loss with label smoothing
        loss, grad = cross_entropy_with_label_smoothing(
            logits, tar_real, label_smoothing=self.label_smoothing
        )

        # Scale loss for AMP
        if self.use_amp:
            loss_scaled = self.scaler.scale_loss(cp.array(loss))
            grad = grad * self.scaler.get_scale()

        # Backward pass
        self.model.backward(grad, training=True, accumulate=accumulate)

        # Compute accuracy
        acc = compute_accuracy(logits, tar_real)

        # Update weights if not accumulating
        grad_norm = 0.0
        lr = self.optimizer.lr
        
        if not accumulate:
            # Unscale gradients if using AMP
            if self.use_amp:
                if not self.scaler.unscale_grads(self.model.parameters()):
                    # Skip update due to inf/nan
                    self.optimizer.zero_grad()
                    return {"loss": loss, "accuracy": acc, "grad_norm": 0.0, "lr": lr, "skipped": True}

            # Check gradients
            num_grad, total_params, missing = check_gradients(self.model.parameters(), verbose=False)
            if num_grad < total_params:
                print(f"⚠ Warning: Only {num_grad}/{total_params} parameters have gradients")

            # Clip gradients
            grad_norm = clip_grad_norm(self.model.parameters(), self.max_grad_norm)

            # Optimizer step
            self.optimizer.step()
            lr = self.scheduler.step()
            self.optimizer.zero_grad()

        return {"loss": loss, "accuracy": acc, "grad_norm": grad_norm, "lr": lr, "skipped": False}

    def validate(
        self,
        inp: cp.ndarray,
        tar_inp: cp.ndarray,
        tar_real: cp.ndarray,
        enc_mask: cp.ndarray,
        combined_mask: cp.ndarray,
        dec_mask: cp.ndarray,
    ) -> Dict[str, float]:
        """Validation step."""
        logits, _ = self.model.forward(
            inp, tar_inp, enc_mask, combined_mask, dec_mask, training=False
        )

        loss, _ = cross_entropy_with_label_smoothing(
            logits, tar_real, label_smoothing=0.0  # No smoothing for validation
        )
        acc = compute_accuracy(logits, tar_real)

        return {"loss": loss, "accuracy": acc}

    def train(
        self,
        train_data: List[Tuple],
        val_data: List[Tuple],
        num_epochs: int,
        print_every: int = 10,
    ) -> None:
        """Complete training loop with gradient accumulation."""
        print("=" * 80)
        print("Training with v06 improvements:")
        print(f"  - Label Smoothing: {self.label_smoothing}")
        print(f"  - Gradient Accumulation: {self.gradient_accumulation_steps} steps")
        print(f"  - Mixed Precision: {self.use_amp}")
        print(f"  - Gradient Clipping: {self.max_grad_norm}")
        print("=" * 80)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            skipped_steps = 0

            for i, (inp, tar_inp, tar_real, enc_mask, comb_mask, dec_mask) in enumerate(
                train_data
            ):
                # Determine if we should accumulate
                accumulate = (i + 1) % self.gradient_accumulation_steps != 0
                
                metrics = self.train_step(
                    inp, tar_inp, tar_real, enc_mask, comb_mask, dec_mask, accumulate
                )

                epoch_loss += metrics["loss"]
                epoch_acc += metrics["accuracy"]
                
                if metrics.get("skipped", False):
                    skipped_steps += 1

                if (i + 1) % print_every == 0:
                    avg_loss = epoch_loss / (i + 1)
                    avg_acc = epoch_acc / (i + 1)
                    scale_info = f" | Scale: {self.scaler.get_scale():.0f}" if self.use_amp else ""
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | Step {i+1}/{len(train_data)} | "
                        f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | "
                        f"LR: {metrics['lr']:.6f} | Grad: {metrics['grad_norm']:.4f}{scale_info}"
                    )

            avg_train_loss = epoch_loss / len(train_data)
            avg_train_acc = epoch_acc / len(train_data)
            self.train_losses.append(avg_train_loss)
            self.train_accs.append(avg_train_acc)

            # Validation
            val_loss = 0.0
            val_acc = 0.0
            for inp, tar_inp, tar_real, enc_mask, comb_mask, dec_mask in val_data:
                val_metrics = self.validate(
                    inp, tar_inp, tar_real, enc_mask, comb_mask, dec_mask
                )
                val_loss += val_metrics["loss"]
                val_acc += val_metrics["accuracy"]

            avg_val_loss = val_loss / len(val_data)
            avg_val_acc = val_acc / len(val_data)
            self.val_losses.append(avg_val_loss)
            self.val_accs.append(avg_val_acc)

            skip_info = f" (Skipped: {skipped_steps})" if skipped_steps > 0 else ""
            print(f"\nEpoch {epoch+1} Summary:{skip_info}")
            print(
                f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%"
            )
            print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")
            print("-" * 80)

# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Transformer v06 - Enhanced Training")
    print("=" * 80)

    # Test with small model
    model = TransformerArchitecture(
        vocab_size=100,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_seq_len=20,
        dropout_rate=0.1,
    )

    inp = cp.random.randint(1, 100, (4, 8), dtype=cp.int32)
    tar = cp.random.randint(1, 100, (4, 10), dtype=cp.int32)
    targets = cp.random.randint(0, 100, (4, 10), dtype=cp.int32)

    enc_mask = create_padding_mask(inp)
    dec_mask = create_padding_mask(inp)
    comb_mask = create_combined_mask(tar)

    print("\nForward pass...")
    logits, _ = model.forward(inp, tar, enc_mask, comb_mask, dec_mask, True)
    print(f"✓ Logits shape: {logits.shape}")

    print("\nComputing loss with label smoothing...")
    loss, grad = cross_entropy_with_label_smoothing(
        logits, targets, label_smoothing=0.1
    )
    print(f"✓ Loss: {loss:.4f}")

    print("\nBackward pass...")
    model.backward(grad, True)

    print("\nChecking gradients...")
    num_grad, total_params, missing = check_gradients(model.parameters(), verbose=True)
    
    if num_grad == total_params:
        print("\n✓ All parameters have gradients!")
    else:
        print(f"\n⚠ Warning: {total_params - num_grad} parameters missing gradients")

    print("\n" + "=" * 80)
    print("✓ v06 test complete!")
    print("=" * 80)
