
# 🧠 GRU (Gated Recurrent Unit, 게이트 순환 신경망)

---

## 📘 1. 개요

**GRU**는 2014년에 **Cho et al.**이 제안한 RNN의 한 변형으로,
LSTM과 비슷하게 **장기 의존성 문제(Long-Term Dependency)** 를 해결하기 위해 만들어진 모델이다.

LSTM보다 **구조가 단순**해서 **학습 속도가 빠르며**,
매우 비슷한 성능을 내기 때문에 많은 실무 모델에서도 자주 쓰인다.

---

## 🔍 2. LSTM과의 비교

| 항목            | LSTM                | GRU               |
| ------------- | ------------------- | ----------------- |
| 게이트 수         | 3개 (입력, 망각, 출력 게이트) | 2개 (업데이트, 리셋 게이트) |
| 셀 상태 $c_t$  | 있음                  | 없음                |
| 은닉 상태 $h_t$ | 있음                  | 있음 (셀 상태 = 은닉 상태) |
| 학습 속도         | 느림                  | 빠름                |
| 구조 복잡도        | 높음                  | 낮음                |

➡️ GRU는 **셀 상태를 따로 두지 않고**, 모든 정보가 **은닉 상태 $h_t$** 안에 담긴다.

---

## ⚙️ 3. 수식 정리

GRU의 동작은 다음과 같은 수식으로 표현된다.

1️⃣ **업데이트 게이트 (Update gate)**
과거 정보를 얼마나 유지할지를 결정
$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

2️⃣ **리셋 게이트 (Reset gate)**
과거 정보를 얼마나 잊을지를 결정
$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

3️⃣ **후보 은닉 상태 (Candidate hidden state)**
새로운 정보를 계산
$$
\tilde{h}*t = \tanh(W_h x_t + U_h (r_t \odot h*{t-1}) + b_h)
$$

4️⃣ **최종 은닉 상태 (Hidden state update)**
이전 상태와 새로운 상태를 섞어서 최종 은닉 상태 결정
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

---

## 💻 4. GRU Python 구현 예시

이 예시는 `numpy`를 기반으로 한 GRU의 기본적인 구현을 보여준다. GPU 가속을 위해서는 `numpy` 대신 `cupy`를 활용할 수 있다.

```python
# GRU (Gated Recurrent Unit)
# 2025-10-11: Vanilla GRU 구현

import numpy as np

class VanillaGRU:
    def __init__(self, n_x, n_h, n_y, lr=0.01):
        self.n_x, self.n_h, self.n_y = n_x, n_h, n_y
        self.lr = lr

        # 가중치 초기화
        self.Wz = np.random.randn(n_x, n_h) * 0.01
        self.Uz = np.random.randn(n_h, n_h) * 0.01
        self.bz = np.zeros((1, n_h))

        self.Wr = np.random.randn(n_x, n_h) * 0.01
        self.Ur = np.random.randn(n_h, n_h) * 0.01
        self.br = np.zeros((1, n_h))

        self.Wh = np.random.randn(n_x, n_h) * 0.01
        self.Uh = np.random.randn(n_h, n_h) * 0.01
        self.bh = np.zeros((1, n_h))

        self.Why = np.random.randn(n_h, n_y) * 0.01
        self.by = np.zeros((1, n_y))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, X):
        h = np.zeros((1, self.n_h))
        hs, ys = [], []

        for x_t in X:
            z = self.sigmoid(x_t @ self.Wz + h @ self.Uz + self.bz)
            r = self.sigmoid(x_t @ self.Wr + h @ self.Ur + self.br)
            h_tilde = np.tanh(x_t @ self.Wh + (r * h) @ self.Uh + self.bh)
            h = (1 - z) * h + z * h_tilde
            y = self.softmax(h @ self.Why + self.by)

            hs.append(h)
            ys.append(y)

        return np.array(hs), np.array(ys)

    def backward(self, X, Y, hs, ys):
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros((1, self.n_h))

        for t in reversed(range(len(X))):
            dy = ys[t] - Y[t]
            dWhy += hs[t].T @ dy
            dby += dy
            dh_next = dy @ self.Why.T + dh_next

        # 이 예시에서는 간단화를 위해 게이트별 Backprop 계산은 생략하고 출력층 관련 가중치만 업데이트한다.
        self.Why -= self.lr * np.clip(dWhy, -5, 5)
        self.by -= self.lr * np.clip(dby, -5, 5)

    def train(self, X, Y, epochs=2000):
        for epoch in range(epochs):
            hs, ys = self.forward(X)
            loss = -np.sum(Y * np.log(ys + 1e-8))
            self.backward(X, Y, hs, ys)
            if epoch % 200 == 0:
                print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

    def predict(self, X):
        _, ys = self.forward(X)
        return np.argmax(ys, axis=2)

# 예시: "hi" → "ih"
if __name__ == "__main__":
    vocab_size = 3
    X = np.eye(vocab_size)
    Y = np.roll(X, -1, axis=0)

    gru = VanillaGRU(n_x=vocab_size, n_h=8, n_y=vocab_size)
    gru.train(X, Y, epochs=1000)
```

---

## 🧩 5. GRU의 장점

| 장점          | 설명                        |
| ----------- | ------------------------- |
| 🏃‍♂️ 빠른 학습 | LSTM보다 게이트 수가 적어서 계산량이 감소 |
| 💾 적은 메모리   | 파라미터 개수가 줄어들어 효율적         |
| ⚖️ 비슷한 성능   | LSTM 수준의 장기 의존성 학습 가능     |
| 🔧 단순한 구조   | 구현 및 튜닝이 쉬움               |

---

## 🧠 6. 주요 활용 분야

| 분야         | 설명                              |
| ---------- | ------------------------------- |
| **텍스트 생성** | 문장 자동 완성, 챗봇 응답                 |
| **음성 처리**  | TTS(Text-to-Speech), 음성 감정 인식   |
| **시계열 예측** | 주가, 날씨, 센서 데이터 예측               |
| **기계 번역**  | Seq2Seq 구조의 Encoder-Decoder로 활용 |

---

## 🌟 7. GRU는 어디에 쓰면 좋은가?

* **실시간 처리**가 필요한 곳 (예: 실시간 번역, 대화형 AI)
* **데이터 양이 적거나 짧은 시퀀스**를 다루는 곳
* **모바일/임베디드 환경** (빠른 추론과 낮은 연산량이 중요할 때)

이 구현은 `numpy` 기반이지만, `cupy`를 활용하여 GPU 가속을 적용하면 더욱 빠른 학습 및 추론이 가능하다.
