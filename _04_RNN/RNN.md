
# 🔁 RNN (Recurrent Neural Network) - 순환 신경망의 개념과 구현

## 1. 개요

RNN(Recurrent Neural Network, 순환 신경망)은 **시퀀스 데이터(Sequence Data)** 를 처리하기 위해 설계된 신경망 구조이다.
입력 간의 **시간적 의존성(temporal dependency)** 을 학습할 수 있으며, 대표적으로 **자연어 처리, 음성 인식, 시계열 예측** 등에서 사용된다.

MLP는 입력을 독립적으로 처리하지만, RNN은 **이전 시점의 출력을 다음 시점의 입력에 반영**함으로써 순환 구조를 가진다.

---

## 2. 구조 및 수식

### 2.1 기본 구조

RNN은 시간 $t$에서의 입력 $x_t$, 은닉 상태 $h_t$, 출력 $y_t$ 로 구성된다.

* 입력: $x_t \in \mathbb{R}^{n_x}$
* 은닉 상태: $h_t \in \mathbb{R}^{n_h}$
* 출력: $y_t \in \mathbb{R}^{n_y}$

시점별 순환 구조는 다음과 같다:

```
x_t ─▶ [RNN Cell] ─▶ h_t ─▶ y_t
           ▲
           │
          h_{t-1}
```

---

### 2.2 순전파 (Forward Propagation)

RNN의 핵심 수식은 다음과 같다:

$$
h_t = f(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

$$
y_t = g(W_{hy} h_t + b_y)
$$

여기서,

* $W_{xh} \in \mathbb{R}^{n_x \times n_h}$ : 입력 → 은닉 가중치
* $W_{hh} \in \mathbb{R}^{n_h \times n_h}$ : 이전 은닉 → 현재 은닉 가중치
* $W_{hy} \in \mathbb{R}^{n_h \times n_y}$ : 은닉 → 출력 가중치
* $b_h$, $b_y$ : 편향
* $f$ : tanh (은닉 활성화 함수)
* $g$ : softmax (출력 활성화 함수)

---

### 2.3 시간 전개 (Unrolled Form)

시간 축으로 펼치면 다음과 같은 형태가 된다:

$$
\begin{align*}
h_1 &= f(W_{xh} x_1 + W_{hh} h_0 + b_h) \\
h_2 &= f(W_{xh} x_2 + W_{hh} h_1 + b_h) \\
\vdots \\
h_T &= f(W_{xh} x_T + W_{hh} h_{T-1} + b_h)
\end{align*}
$$

출력은 각 시점별로:

$$
y_t = g(W_{hy} h_t + b_y)
$$

---

## 3. 손실 함수

시퀀스 전체의 손실은 모든 시점의 손실 평균으로 계산된다.

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{j=1}^{C} y_{tj} \log(\hat{y}_{tj})
$$

---

## 4. 역전파 (BPTT: Backpropagation Through Time)

RNN의 학습은 **시간 역전파(Backpropagation Through Time, BPTT)** 알고리즘으로 수행된다.
기본 아이디어는 MLP의 역전파를 “시간 축”으로 확장하는 것이다.

### 4.1 기본 수식

출력층 오차:

$$
\delta^{(y)}_t = \hat{y}_t - y_t
$$

은닉층 오차 (역전파):

$$
\delta^{(h)}*t = (\delta^{(y)}*t W*{hy}^\top + \delta^{(h)}*{t+1} W_{hh}^\top) \odot f'(h_t)
$$

여기서 $\odot$ 는 요소별 곱(Hadamard product)이다.

### 4.2 기울기 누적

모든 시점의 기울기를 누적하여 가중치 업데이트에 사용하며, 기울기 폭주(exploding gradients)를 방지하기 위해 기울기 클리핑(gradient clipping)을 적용한다:

$$
\frac{\partial \mathcal{L}}{\partial W_{xh}} = \sum_{t=1}^{T} x_t^\top \delta^{(h)}*t \\
\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} h_{t-1}^\top \delta^{(h)}*t \\
\frac{\partial \mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T} h_t^\top \delta^{(y)}_t
$$

---

## 5. 구현 코드 설명

### 5.1 클래스 초기화

```python
class VanillaRNN:
    def __init__(self, n_x, n_h, n_y, lr=0.01):
        self.Wxh = np.random.randn(n_x, n_h) * 0.01
        self.Whh = np.random.randn(n_h, n_h) * 0.01
        self.Why = np.random.randn(n_h, n_y) * 0.01
        self.bh = np.zeros((1, n_h))
        self.by = np.zeros((1, n_y))
        self.lr = lr
```

---

### 5.2 순전파 (Forward Pass)

```python
def forward(self, X):
    h, hs, ys = np.zeros((1, self.Whh.shape[0])), [], []
    for x_t in X:
        h = np.tanh(x_t @ self.Wxh + h @ self.Whh + self.bh)
        y = self.softmax(h @ self.Why + self.by)
        hs.append(h)
        ys.append(y)
    return np.array(hs), np.array(ys)
```

---

### 5.3 역전파 (BPTT)

```python
def backward(self, X, Y, hs, ys):
    dWxh = np.zeros_like(self.Wxh)
    dWhh = np.zeros_like(self.Whh)
    dWhy = np.zeros_like(self.Why)
    dbh = np.zeros_like(self.bh)
    dby = np.zeros_like(self.by)

    dh_next = np.zeros_like(hs[0])

    for t in reversed(range(len(X))):
        dy = ys[t] - Y[t]
        dWhy += hs[t].T @ dy
        dby += dy
        dh = dy @ self.Why.T + dh_next
        dh_raw = (1 - hs[t]**2) * dh
        dbh += dh_raw
        dWxh += X[t].T @ dh_raw
        dWhh += hs[t-1].T @ dh_raw if t > 0 else 0
        dh_next = dh_raw @ self.Whh.T

    # Update
    for param, dparam in zip(
        [self.Wxh, self.Whh, self.Why, self.bh, self.by],
        [dWxh, dWhh, dWhy, dbh, dby]
    ):
        param -= self.lr * np.clip(dparam, -5, 5)
```

---

### 5.4 학습 및 예측 유틸리티 함수

*   `compute_loss(Y, ys)`: 실제 정답 `Y`와 모델의 예측 `ys`를 기반으로 교차 엔트로피 손실을 계산한다.
*   `prepare_sequences(words, char_to_idx)`: 주어진 단어 리스트를 문자 단위의 원-핫 인코딩 시퀀스로 변환하여 학습 데이터(`X_train`, `Y_train`)를 준비한다.
*   `train_words(words, char_to_idx, epochs, print_interval)`: `prepare_sequences`를 호출하여 학습 데이터를 준비한 후, `train` 메서드를 사용하여 모델을 학습시키는 편리한 래퍼 함수이다.
*   `predict(seed_text, char_to_idx, idx_to_char, length)`: 주어진 `seed_text`를 시작으로 RNN 모델을 사용하여 `length`만큼 다음 문자를 예측하여 시퀀스를 생성한다. 예측 과정에서 이전 예측 문자가 다음 시점의 입력으로 사용된다.

## 6. 예제: 문자 단위 RNN (Character-Level RNN)

```python
# 입력: ["h", "e", "l", "l", "o"]
# 출력: ["e", "l", "l", "o", " "]
X = np.eye(5)  # One-hot encoding
Y = np.roll(X, -1, axis=0)

rnn = VanillaRNN(n_x=5, n_h=8, n_y=5)

for epoch in range(1000):
    hs, ys = rnn.forward(X)
    loss = -np.sum(Y * np.log(ys + 1e-8))
    rnn.backward(X, Y, hs, ys)
```

---

## 7. 시각화

* 은닉 상태의 변화를 t-SNE 등으로 시각화하면 **시퀀스 내 문맥적 패턴**을 관찰할 수 있다.
* 출력 확률을 heatmap으로 표현하면 모델의 “예측 분포”를 직관적으로 이해할 수 있다.

---

## 8. 결론

Vanilla RNN은 가장 기본적인 순환 신경망으로, `cupy`를 활용하여 GPU 가속을 지원한다.
시퀀스 데이터를 처리할 수 있으나 **장기 의존성 문제(Long-Term Dependency)** 로 인해
긴 문맥을 기억하는 데 어려움이 있다.

이 한계를 극복하기 위해 이후에 **LSTM**과 **GRU**가 등장한다.

---

### 🔗 다음 단계

| 모델        | 특징                          |
| --------- | --------------------------- |
| LSTM      | 게이트 구조로 장기 의존성 해결           |
| GRU       | LSTM보다 간단한 구조로 유사 성능        |
| BiRNN     | 양방향 문맥 정보 학습                |
| Seq2Seq   | 인코더-디코더 구조, 번역/요약 등 응용      |
| Attention | 선택적 정보 집중 (Transformer의 기초) |
