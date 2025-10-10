
# LSTM (Long Short-Term Memory, 장기 기억 순환 신경망)

### 1. 개요

LSTM은 전통적인 RNN의 **장기 의존성 문제**(Long-term Dependency)를 해결하기 위해 고안된 순환 신경망 구조입니다.
Vanilla RNN은 hidden state $h_t$를 통해 이전 정보를 전달하지만, 시퀀스가 길어지면 **기울기 소실(vanishing gradient) 문제**가 발생하여 먼 과거 정보가 제대로 반영되지 않습니다.

LSTM은 셀 상태(cell state) $C_t$를 도입하고 **게이트 구조**를 통해 정보를 선택적으로 기억하거나 잊음으로써 이 문제를 해결합니다.

---

### 2. 구성 요소

LSTM의 핵심은 **게이트(gate) 구조**입니다. 각 게이트는 0~1 사이의 값으로 정보를 얼마나 통과시킬지 결정합니다. 입력 데이터는 `W_embed`를 통해 임베딩 벡터로 변환되어 LSTM 셀로 전달된다.

1. **입력 게이트(Input Gate, $i_t$)**

   * 현재 입력 $x_t$와 이전 hidden state $h_{t-1}$를 기반으로 새 정보를 얼마나 셀 상태에 반영할지 결정합니다.
   * 수식:
     $$
     i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
     $$

2. **망각 게이트(Forget Gate, $f_t$)**

   * 이전 셀 상태 (C_{t-1}) 중 어떤 정보를 잊을지 결정합니다.
   * 수식:
     $$
     f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
     $$

3. **셀 후보 상태(Cell Candidate, $\tilde{C}_t$)**

   * 새로 추가될 후보 정보
   * 수식:
     $$
     \tilde{C}*t = \tanh(W_c x_t + U_c h*{t-1} + b_c)
     $$

4. **셀 상태 업데이트(Cell State, $C_t$)**

   * 망각 게이트와 입력 게이트를 반영하여 업데이트
   * 수식:
     $$
     C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
     $$
   * 여기서 $\odot$는 원소별 곱(Element-wise multiplication)

5. **출력 게이트(Output Gate, $o_t$)**

   * 최종 hidden state $h_t$를 결정
   * 수식:
     $$
     o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
     $$
     $$
     h_t = o_t \odot \tanh(C_t)
     $$

---

### 3. 장점

* **장기 의존성 학습 가능**
  → 이전 정보가 멀리 있어도 기억 가능
* **기울기 소실 문제 완화**
* **문자 시퀀스, 음성, 시계열 데이터**에 강력

---

### 4. Vanilla RNN vs LSTM 비교

| 항목     | Vanilla RNN        | LSTM                                  |
| ------ | ------------------ | ------------------------------------- |
| 기억 장치  | Hidden state (h_t) | Hidden state (h_t) + Cell state (C_t) |
| 장기 의존성 | 어려움                | 가능                                    |
| 구조     | 단순                 | 복잡 (4개의 게이트)                          |
| 계산량    | 낮음                 | 높음                                    |

---

### 5. 구현 상세 (CuPy 기반)

*   **CuPy Fused Kernels**: `_fused_forward_cell`과 `_fused_backward_cell`과 같은 CuPy 퓨즈드 커널을 사용하여 순전파 및 역전파 계산의 성능을 최적화한다.
*   **가중치 초기화**: `W_embed`, `W_x`, `W_h`, `Why` 등 모든 가중치에 대해 Xavier 초기화를 적용하여 학습 안정성을 높인다.
*   **Adam 옵티마이저**: 모델 파라미터(`W_embed`, `W_x`, `W_h`, `b`, `Why`, `by`)는 Adam 옵티마이저를 통해 갱신된다. Adam의 모멘트(`m`, `v`)와 시간 스텝(`t`)은 모델과 함께 관리된다.
*   **배치 처리**: `forward` 및 `backward` 메서드는 여러 시퀀스를 동시에 처리하는 배치 처리 방식으로 구현되어 GPU 활용 효율을 극대화한다.
*   **순전파(forward) 시**:
    1.  입력 인덱스를 `W_embed`를 통해 임베딩 벡터로 변환한다.
    2.  각 시점의 입력 $x_t$와 이전 hidden state $h_{t-1}$를 기반으로 게이트 값($i_t, f_t, o_t, g_t$)을 계산한다.
    3.  `_fused_forward_cell`을 사용하여 셀 상태 $C_t$와 hidden state $h_t$를 업데이트한다.
    4.  최종 hidden state $h_t$를 사용하여 출력 $y_t = \text{softmax}(h_t W_{hy} + b_y)$를 계산한다.
*   **역전파(backward) 시**:
    1.  출력층의 오차부터 시작하여 각 게이트별 그라디언트를 계산한다.
    2.  `_fused_backward_cell`을 사용하여 셀 상태 $C_t$를 통해 기울기 흐름을 유지하며 역전파를 수행한다.
    3.  **기울기 클리핑**: 기울기 폭주(exploding gradients)를 방지하기 위해 계산된 그라디언트에 클리핑을 적용한다.

---

### 6. 학습 & 예측

*   **학습 데이터**: `create_batches_for_embedding` 함수를 통해 준비된 배치 단위의 시퀀스 데이터.
*   **손실 함수**: **Cross-entropy**를 사용하여 모델의 예측과 실제 정답 간의 차이를 측정한다.
*   **최적화**: **Adam 옵티마이저**를 사용하여 모델의 모든 파라미터(임베딩, 게이트 가중치, 출력 가중치 등)를 효율적으로 갱신한다.
*   **학습률 스케줄링**: 500 에포크마다 학습률을 절반으로 감소시켜 학습의 안정성과 성능 향상을 도모한다.
*   **체크포인트 및 학습 재개**:
    *   `save_every` 파라미터에 따라 정기적으로 모델의 모든 파라미터(가중치, Adam 옵티마이저의 모멘트 값, 현재 에포크)를 `.npz` 파일로 저장한다.
    *   `load_model` 함수를 통해 저장된 체크포인트로부터 학습을 정확히 재개할 수 있으며, 이때 Adam 옵티마이저의 상태도 함께 복원된다.
*   **조기 종료 (Early Stopping)**: `target_loss`에 도달하면 학습을 조기에 종료하여 과적합을 방지하고 효율적인 학습을 유도한다.
*   **예측**: `predict` 함수는 `seed_text`를 기반으로 다음 문자를 순차적으로 생성한다. 이때 `item_to_idx`, `idx_to_item`을 사용하여 문자와 인덱스 간 변환을 처리하며, `max_len`과 `end_idx`를 통해 예측 길이와 종료 조건을 제어한다.

---

### 7. 실용 팁

* 은닉층 크기(hidden_size) 32~128 정도 추천
* 학습률(lr)은 Vanilla RNN보다 조금 낮춰서 안정화
* CuPy/NumPy 기반으로 구현 후 GPU 학습 가능
* 여러 단어 학습 시 **train_words()** 패턴 그대로 사용 가능

---

### 8. 데이터 준비 (create_batches_for_embedding)

`create_batches_for_embedding` 함수는 LSTM 모델 학습을 위한 데이터를 효율적으로 준비하는 유틸리티이다.

*   **시퀀스 인덱싱**: 입력 시퀀스의 각 항목(문자 등)을 정수 인덱스로 변환한다.
*   **길이별 정렬**: 배치 처리를 위해 시퀀스 길이에 따라 정렬한다.
*   **패딩**: 배치 내 모든 시퀀스의 길이를 가장 긴 시퀀스에 맞춰 패딩한다.
*   **원-핫 인코딩**: 타겟 시퀀스를 원-핫 인코딩 형식으로 변환한다.
*   **배치 생성**: 위 과정을 거쳐 `(X_batch_idx, Y_batch)` 형태의 CuPy 배열 배치를 생성한다.

---

### 9. 결론

LSTM은 RNN의 장기 의존성 문제를 효과적으로 해결하는 강력한 시퀀스 모델이다. `LSTM_v05.py` 구현은 CuPy 퓨즈드 커널을 활용한 GPU 가속, Adam 옵티마이저, Xavier 초기화, 배치 처리, 학습률 스케줄링, 체크포인트 및 재개, 조기 종료 등 최신 딥러닝 학습 기법들을 통합하여 높은 성능과 안정성을 제공한다.
