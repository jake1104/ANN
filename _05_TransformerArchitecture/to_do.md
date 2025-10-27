v08 TODO
- Cupy RawKernel로 FeedForward + Softmax Fused (`cupy.RawKernel`로 GEMM + ReLU + Bias 통합)
- FlashAttention 커널 직접 구현 (Cupy Custom Kernel) (`exp(QKᵀ / sqrt(d))`와 Softmax+matmul을 한 번에)
- Fused LayerNorm (`cupyx.fuse()` 가능)
- Asynchronous Streams (Overlapping) (forward/backward 병렬화)
- Checkpointing 비활성화 + FP16 완전 일관성 확보 (불필요한 cast 줄이기)
- CuPy Memory Pool 고정 (메모리 재할당 방지)

V09 TODO
- LayerNorm 커널화 (backward 포함 커널 작성 (RawKernel))
- FNN Fused kernel (matmul+relu+dropout 한 번에)
- QKC Cache 유지 (encoder와 decoder에서 공유)
- Async Stream overlap (encoder/decoder 병렬 실행)
- CUDA Graphs (반복 커널 호출 최소화)
- FP16 tensor reuse (cast 최소화)
