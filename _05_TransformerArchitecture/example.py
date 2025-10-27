import cupy as cp
import numpy as np
from typing import List, Tuple, Dict
import time
import os
import argparse

# Import from v07 implementation
from .TransformerArchitecture_v07 import (
    TransformerArchitecture,
    AdamW,
    WarmupCosineScheduler,
    Parameter,
    GlobalConfig,
    create_padding_mask,
    create_look_ahead_mask,
    create_combined_mask,
    cross_entropy_with_label_smoothing,
    clip_grad_norm,
    check_gradients,
    ModelSaver,
    compute_accuracy,
)

# ============================================================================
# Custom Dataloader for NMT - OPTIMIZED
# ============================================================================


class CustomBatch:
    """A simple batch object that holds data for various inputs."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class CustomDataLoader:
    """Optimized DataLoader with NumPy shuffle."""

    def __init__(self, batch_size, shuffle=True, **kwargs):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.datasets = {k: v for k, v in kwargs.items() if v is not None}
        self.keys = list(self.datasets.keys())
        self.num_samples = len(self.datasets[self.keys[0]])
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Optimized shuffle using NumPy
        if self.shuffle:
            indices = cp.asarray(np.random.permutation(self.num_samples))
        else:
            indices = cp.arange(self.num_samples)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_data = {key: self.datasets[key][batch_indices] for key in self.keys}
            yield CustomBatch(**batch_data)

    def __len__(self):
        return self.num_batches


# ============================================================================
# Data Loading and Preprocessing for NMT
# ============================================================================


def load_translation_pairs(filepath: str, num_pairs: int) -> List[Tuple[str, str]]:
    """Load English-Korean sentence pairs from the dataset."""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_pairs:
                break
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def build_unified_vocab(
    all_sentences: List[str],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build a single unified vocabulary from a list of sentences."""
    chars = set()
    for sentence in all_sentences:
        for char in sentence:
            chars.add(char)

    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for char in sorted(list(chars)):
        if char not in vocab:
            vocab[char] = len(vocab)

    id2word = {v: k for k, v in vocab.items()}
    return vocab, id2word


def tokenize_and_pad(
    sentences: List[str], vocab: Dict[str, int], max_len: int
) -> cp.ndarray:
    """Tokenize sentences using a unified vocab, add SOS/EOS, and pad."""
    tokenized_list = []
    for sentence in sentences:
        tokens = (
            [vocab["<sos>"]]
            + [vocab.get(char, vocab["<unk>"]) for char in sentence]
            + [vocab["<eos>"]]
        )
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([vocab["<pad>"]] * (max_len - len(tokens)))
        tokenized_list.append(tokens)
    return cp.array(tokenized_list, dtype=cp.int32)


# ============================================================================
# Configuration - v07 OPTIMIZED
# ============================================================================


class Config:
    # Model architecture
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 60  # Reduced from 72
    dropout_rate: float = 0.1

    # Training data
    batch_size: int = 128
    num_sentence_pairs: int = 5000

    # Training hyperparameters (v07 optimized)
    num_epochs: int = 30
    learning_rate: float = 0.0003  # 3e-4
    warmup_ratio: float = 0.05  # Increased from 0.01
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05  # Reduced from 0.1

    # Advanced training features
    gradient_accumulation_steps: int = 2
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False  # Only for large models

    # Logging
    print_every: int = 20
    save_every_epoch: int = 5

    # Computed
    total_steps: int = 1
    warmup_steps: int = 1


# ============================================================================
# Utility & Inference
# ============================================================================


def tokens_to_text(tokens: cp.ndarray, id2word: Dict[int, str]) -> str:
    """Converts a sequence of tokens back to a string."""
    s = []
    for t in tokens:
        token_id = int(t)
        if token_id == 0 or token_id == 2:  # <pad> or <eos>
            break
        if token_id != 1:  # skip <sos>
            s.append(id2word.get(token_id, "?"))
    return "".join(s)


def translate(
    model: TransformerArchitecture,
    sentence: str,
    config: Config,
    vocab: Dict,
    id2word: Dict,
) -> str:
    """Translate a source sentence using greedy decoding."""
    # Tokenize source sentence
    src_tokens = [vocab.get(c, vocab["<unk>"]) for c in sentence]
    src_tokens = [vocab["<sos>"]] + src_tokens + [vocab["<eos>"]]

    # Pad to max length
    if len(src_tokens) < config.max_seq_len:
        src_tokens.extend([vocab["<pad>"]] * (config.max_seq_len - len(src_tokens)))
    else:
        src_tokens = src_tokens[: config.max_seq_len]

    enc_input = cp.array([src_tokens], dtype=cp.int32)
    enc_padding_mask = create_padding_mask(enc_input)

    # Start with <sos> token
    dec_input_tokens = [vocab["<sos>"]]

    for _ in range(config.max_seq_len - 1):
        dec_input = cp.array(
            [
                dec_input_tokens
                + [vocab["<pad>"]] * (config.max_seq_len - len(dec_input_tokens))
            ],
            dtype=cp.int32,
        )

        # Vectorized mask creation
        combined_mask = create_combined_mask(dec_input)

        logits, _ = model.forward(
            enc_input,
            dec_input,
            enc_padding_mask,
            combined_mask,
            enc_padding_mask,
            training=False,
        )

        # Get next token
        next_token_logits = logits[0, len(dec_input_tokens) - 1, :]
        next_token = int(cp.argmax(next_token_logits))

        # Stop if <eos> or <pad>
        if next_token == vocab["<eos>"] or next_token == vocab["<pad>"]:
            break

        dec_input_tokens.append(next_token)

    return tokens_to_text(cp.array(dec_input_tokens[1:]), id2word)


# ============================================================================
# Main
# ============================================================================


def main():
    usage_message = """
    ================================================================================
    Transformer v07 NMT (kor-eng) 사용법 - OPTIMIZED VERSION
    ================================================================================
    v07의 주요 개선사항:
    - Vectorized Mask Creation: 반복문 제거로 속도 향상
    - Proper Mixed Precision: FP16 연산 전체 적용
    - Optimized Gradient Zeroing: set_to_none=True로 메모리 효율 개선
    - NumPy Shuffle: CuPy random보다 빠른 permutation 사용
    - QKV Fusion: 단일 matmul로 Q, K, V 동시 계산
    - CuPy Memory Pool: 메모리 할당 최적화
    - Reduced Sequence Length: 72→60으로 메모리 절약
    - Improved Hyperparameters: warmup_ratio=0.05, label_smoothing=0.05
    
    사용 가능한 인자:
      --force-train      : 저장된 모델을 무시하고 처음부터 새로 학습
      --epochs <숫자>     : 학습할 에포크 수 (기본값: 30)
      --batch-size <숫자> : 배치 크기 (기본값: 128)
      --lr <숫자>         : 학습률 (기본값: 0.0003)
      --num-pairs <숫자>  : 학습에 사용할 문장 쌍의 수 (기본값: 5000)
      --label-smoothing <숫자> : Label smoothing factor (기본값: 0.05)
      --grad-accum <숫자> : Gradient accumulation steps (기본값: 2)
      --warmup-ratio <숫자> : Warmup ratio (기본값: 0.05)
      --no-mixed-precision : Mixed precision 비활성화
      --load-path <경로>  : 지정된 경로에서 모델을 불러와 학습 재개
      --save-path <경로>  : 학습 완료 후 모델을 지정된 경로에 저장

    예시:
      # 모델을 불러와 10 에포크 추가 학습 (v07 기능 활용)
      python -m ANN._05_TransformerArchitecture.example_v07 --epochs 10

      # 처음부터 학습 (최적화된 하이퍼파라미터)
      python -m ANN._05_TransformerArchitecture.example_v07 --force-train --epochs 30
      
      # Mixed precision 없이 학습
      python -m ANN._05_TransformerArchitecture.example_v07 --no-mixed-precision --epochs 20
    ================================================================================
    """
    print(usage_message)

    parser = argparse.ArgumentParser(description="Train a Transformer (v07) for NMT.")
    parser.add_argument(
        "--force-train", action="store_true", help="Force retraining from scratch."
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate.")
    parser.add_argument(
        "--num-pairs", type=int, default=5000, help="Number of sentence pairs."
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.05, help="Label smoothing factor."
    )
    parser.add_argument(
        "--grad-accum", type=int, default=2, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.05, help="Warmup ratio."
    )
    parser.add_argument(
        "--no-mixed-precision", action="store_true", help="Disable mixed precision."
    )
    parser.add_argument(
        "--load-path", type=str, default=None, help="Path to load model from."
    )
    parser.add_argument(
        "--save-path", type=str, default=None, help="Path to save final model."
    )
    args = parser.parse_args()

    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.num_sentence_pairs = args.num_pairs
    config.label_smoothing = args.label_smoothing
    config.gradient_accumulation_steps = args.grad_accum
    config.warmup_ratio = args.warmup_ratio
    config.use_mixed_precision = not args.no_mixed_precision

    # Set global config
    GlobalConfig.USE_MIXED_PRECISION = config.use_mixed_precision
    GlobalConfig.GRADIENT_CHECKPOINTING = config.gradient_checkpointing

    print("\n" + "=" * 80)
    print("v07 Configuration (OPTIMIZED):")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(
        f"  Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}"
    )
    print(f"  Label Smoothing: {config.label_smoothing} (reduced)")
    print(f"  Warmup Ratio: {config.warmup_ratio} (increased)")
    print(f"  Mixed Precision: {config.use_mixed_precision}")
    print(f"  Max Sequence Length: {config.max_seq_len} (reduced)")
    print(f"  Dropout Rate: {config.dropout_rate}")
    print("=" * 80)

    print("\nLoading and preprocessing data...")
    data_path = os.path.join(os.path.dirname(__file__), "kor-eng", "kor.txt")
    pairs = load_translation_pairs(data_path, num_pairs=config.num_sentence_pairs)

    src_sents = [pair[0] for pair in pairs]
    tgt_sents = [pair[1] for pair in pairs]

    vocab, id2word = build_unified_vocab(src_sents + tgt_sents)
    print(f"✓ Unified vocab created: {len(vocab)} chars")

    src_tokenized = tokenize_and_pad(src_sents, vocab, config.max_seq_len)
    tgt_tokenized = tokenize_and_pad(tgt_sents, vocab, config.max_seq_len)
    print(f"✓ Data tokenized and padded to max_seq_len={config.max_seq_len}")

    # Optimized shuffle using NumPy
    indices = cp.asarray(np.random.permutation(len(pairs)))
    val_split = int(0.1 * len(indices))
    train_indices, val_indices = indices[:-val_split], indices[-val_split:]

    train_src, val_src = src_tokenized[train_indices], src_tokenized[val_indices]
    train_tgt, val_tgt = tgt_tokenized[train_indices], tgt_tokenized[val_indices]

    # Decoder inputs and labels
    train_dec_inputs, train_labels = train_tgt[:, :-1], train_tgt[:, 1:]

    num_train_batches = (len(train_src) + config.batch_size - 1) // config.batch_size
    config.total_steps = num_train_batches * config.num_epochs
    config.warmup_steps = int(config.warmup_ratio * config.total_steps)

    train_loader = CustomDataLoader(
        enc_inputs=train_src,
        dec_inputs=train_dec_inputs,
        labels=train_labels,
        batch_size=config.batch_size,
        shuffle=True,
    )
    print(f"✓ Dataloaders created. {num_train_batches} training batches per epoch.")
    print(
        f"✓ Warmup steps: {config.warmup_steps} ({config.warmup_ratio*100:.1f}% of {config.total_steps} total steps)"
    )

    print("\nInitializing model...")
    model = TransformerArchitecture(
        vocab_size=len(vocab),
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout_rate=config.dropout_rate,
    )
    print(
        f"✓ Model initialized ({sum(p.data.size for p in model.parameters()):,} params)"
    )

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_nmt_v07")
    os.makedirs(checkpoint_dir, exist_ok=True)

    default_model_path = os.path.join(checkpoint_dir, "nmt_model_v07.npz")
    load_path = args.load_path if args.load_path else default_model_path
    save_path = args.save_path if args.save_path else default_model_path

    start_epoch = 0
    if not args.force_train and os.path.exists(load_path):
        print(f"\nLoading model from {load_path}...")
        try:
            start_epoch = model.load(load_path)
            if start_epoch is not None:
                print(f"✓ Resuming training from epoch {start_epoch + 1}")
            else:
                print(f"✓ Model loaded, starting from epoch 1")
                start_epoch = 0
        except Exception as e:
            print(f"⚠ Error loading model: {e}. Starting from scratch.")
            start_epoch = 0
    else:
        if args.force_train:
            print("\n✓ Force training enabled. Starting from scratch.")
        else:
            print("\n✓ No saved model found. Starting training from scratch.")

    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=config.warmup_steps, total_steps=config.total_steps
    )
    print("✓ Optimizer and scheduler ready")
    print(
        f"  - Optimizer: AdamW (lr={config.learning_rate}, weight_decay={config.weight_decay})"
    )
    print(
        f"  - Scheduler: Warmup Cosine (warmup={config.warmup_steps}, total={config.total_steps})"
    )

    print("\n" + "=" * 80)
    print("Starting training with v07 optimizations...")
    print("=" * 80 + "\n")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        total_loss, total_acc = 0.0, 0.0
        skipped_updates = 0

        for step, batch in enumerate(train_loader, start=1):
            # Vectorized mask creation
            enc_padding_mask = create_padding_mask(batch.enc_inputs)
            combined_mask = create_combined_mask(batch.dec_inputs)

            # Determine if accumulating gradients
            is_accumulating = step % config.gradient_accumulation_steps != 0

            # Forward pass
            logits, _ = model.forward(
                batch.enc_inputs,
                batch.dec_inputs,
                enc_padding_mask,
                combined_mask,
                enc_padding_mask,
                training=True,
            )

            # Compute loss with label smoothing
            loss, grad = cross_entropy_with_label_smoothing(
                logits, batch.labels, label_smoothing=config.label_smoothing
            )

            # Backward pass
            model.backward(grad, training=True, accumulate=is_accumulating)

            # Update weights if not accumulating
            if not is_accumulating:
                # Check gradients periodically
                if step == 1 or step % 100 == 0:
                    num_grad, total_params, missing = check_gradients(
                        model.parameters(), verbose=False
                    )
                    if num_grad < total_params:
                        print(
                            f"⚠ Step {step}: {num_grad}/{total_params} parameters have gradients"
                        )

                # Gradient clipping
                grad_norm = clip_grad_norm(model.parameters(), config.max_grad_norm)

                # Check for invalid gradients
                has_invalid = False
                for param in model.parameters():
                    if param.grad is not None:
                        if cp.any(cp.isnan(param.grad)) or cp.any(cp.isinf(param.grad)):
                            has_invalid = True
                            break

                if has_invalid:
                    print(f"⚠ Step {step}: Invalid gradients detected, skipping update")
                    optimizer.zero_grad(set_to_none=True)
                    skipped_updates += 1
                else:
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            # Compute accuracy
            preds = cp.argmax(logits, axis=-1)
            acc = compute_accuracy(logits, batch.labels)
            total_loss += float(loss)
            total_acc += acc

            if step % config.print_every == 0:
                avg_loss = total_loss / step
                avg_acc = total_acc / step
                current_lr = scheduler.optimizer.lr
                print(
                    f"Epoch {epoch+1}/{config.num_epochs} | "
                    f"Step {step}/{num_train_batches} | "
                    f"Loss: {loss:.4f} | "
                    f"Acc: {acc:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )

        avg_loss = total_loss / num_train_batches
        avg_acc = total_acc / num_train_batches

        skip_info = (
            f" (Skipped updates: {skipped_updates})" if skipped_updates > 0 else ""
        )
        print(
            f"\n→ Epoch {epoch+1} done | Avg Train Loss: {avg_loss:.4f} | Avg Train Acc: {avg_acc:.2f}%{skip_info}\n"
        )

        # Save checkpoint
        if (epoch + 1) % config.save_every_epoch == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"nmt_model_v07_epoch_{epoch+1}.npz"
            )
            model.save(checkpoint_path, epoch=epoch + 1)
            print(f"✓ Checkpoint saved: {checkpoint_path}")

    print(f"\nSaving final model to {save_path}...")
    model.save(save_path, epoch=config.num_epochs)

    print("\n" + "=" * 80)
    print("Validating and showing translation examples...")
    print("=" * 80 + "\n")

    # Validation examples
    val_src_sents = [tokens_to_text(s, id2word) for s in val_src[:10]]
    val_tgt_sents = [tokens_to_text(s, id2word) for s in val_tgt[:10]]

    print("Translation Examples (Validation Set):\n")
    for i in range(min(5, len(val_src_sents))):
        src_sent = val_src_sents[i]
        tgt_sent = val_tgt_sents[i]

        translated_sent = translate(model, src_sent, config, vocab, id2word)

        print("-" * 80)
        print(f"Source:      {src_sent}")
        print(f"Target:      {tgt_sent}")
        print(f"Predicted:   {translated_sent}")
        print()

    # Compute validation metrics
    print("\nComputing validation metrics...")
    val_loss_total = 0.0
    val_acc_total = 0.0
    val_samples = 0

    val_batch_size = 32
    for i in range(0, len(val_src), val_batch_size):
        end_idx = min(i + val_batch_size, len(val_src))
        batch_src = val_src[i:end_idx]
        batch_tgt = val_tgt[i:end_idx]

        batch_dec_inputs = batch_tgt[:, :-1]
        batch_labels = batch_tgt[:, 1:]

        enc_mask = create_padding_mask(batch_src)
        combined_mask = create_combined_mask(batch_dec_inputs)

        logits, _ = model.forward(
            batch_src,
            batch_dec_inputs,
            enc_mask,
            combined_mask,
            enc_mask,
            training=False,
        )

        loss, _ = cross_entropy_with_label_smoothing(
            logits, batch_labels, label_smoothing=0.0
        )
        acc = compute_accuracy(logits, batch_labels)

        val_loss_total += loss * (end_idx - i)
        val_acc_total += acc * (end_idx - i)
        val_samples += end_idx - i

    val_loss_avg = val_loss_total / val_samples
    val_acc_avg = val_acc_total / val_samples

    print(f"\nValidation Results:")
    print(f"  Loss: {val_loss_avg:.4f}")
    print(f"  Accuracy: {val_acc_avg:.2f}%")

    print("\n" + "=" * 80)
    print("✓ Training complete!")
    print("\nv07 Optimizations Applied:")
    print(f"  ✓ Vectorized Mask Creation: No loops")
    print(f"  ✓ Proper Mixed Precision: {config.use_mixed_precision}")
    print(f"  ✓ Gradient Zeroing: set_to_none=True")
    print(f"  ✓ Optimized Shuffle: NumPy permutation")
    print(f"  ✓ QKV Fusion: Single matmul")
    print(f"  ✓ CuPy Memory Pool: Enabled")
    print(f"  ✓ Label Smoothing: {config.label_smoothing} (optimized)")
    print(f"  ✓ Warmup Ratio: {config.warmup_ratio} (increased)")
    print(f"  ✓ Gradient Accumulation: {config.gradient_accumulation_steps} steps")
    print(f"  ✓ Max Seq Length: {config.max_seq_len} (reduced)")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
