import numpy as np
import pandas as pd
import yfinance as yf
import gym
from gym import spaces
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.preprocessing import StandardScaler

# 데이터 로드 및 기술적 지표 계산
df = pd.read_csv('./000660.csv', encoding='cp949')
df.rename(columns={'종가':'Close'}, inplace=True)
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['RSI'] = 100 - (100 / (1 + df['Close'].diff().apply(lambda x: np.maximum(x, 0))
                          .rolling(window=14).mean() / df['Close'].diff()
                          .apply(lambda x: np.abs(np.minimum(x, 0)))
                          .rolling(window=14).mean()))
df = df.dropna().reset_index()

# 입력 데이터 정규화
scaler = StandardScaler()
feature_cols = ['Close', 'MA10', 'MA50', 'RSI']
scaler.fit(df[feature_cols])

# 주식 트레이딩 환경 정의
class StockTradingEnv(gym.Env):
    def __init__(self, df, scaler):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.scaler = scaler
        self.max_steps = len(df) - 1
        self.current_step = 0

        # 상태 공간: [현재 가격, MA10, MA50, RSI, 보유 여부]
        self.action_space = spaces.Discrete(3)  # 매도(0), 보유(1), 매수(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        # 초기 자본금 및 포지션
        self.initial_balance = 10000000
        self.balance = self.initial_balance
        self.position = 0  # 0: 현금, 1: 주식 보유
        self.asset_value = self.initial_balance  # 포트폴리오 총 가치

        # 추적을 위한 리스트
        self.balance_history = []
        self.asset_value_history = []
        self.position_history = []
        self.action_history = []
        self.price_history = []

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.asset_value = self.initial_balance
        self.current_step = 0

        # 히스토리 초기화
        self.balance_history = [self.balance]
        self.asset_value_history = [self.asset_value]
        self.position_history = [self.position]
        self.action_history = []
        self.price_history = [self.df.loc[self.current_step, 'Close']]

        return self._next_observation()

    def _next_observation(self):
        raw_state = self.df.loc[self.current_step, ['Close', 'MA10', 'MA50', 'RSI']]
        raw_state_df = pd.DataFrame([raw_state])
        scaled_state = self.scaler.transform(raw_state_df)[0]
        frame = np.append(scaled_state, self.position)
        return frame.astype(np.float32)

    def step(self, action):
        done = False
        price = self.df.loc[self.current_step, 'Close']

        # 이전 포트폴리오 가치
        prev_asset_value = self.asset_value

        # 행동에 따른 포트폴리오 업데이트
        if action == 0:  # 매도
            if self.position == 1:
                self.balance += price
                self.position = 0
        elif action == 1:  # 보유
            pass
        elif action == 2:  # 매수
            if self.position == 0:
                self.balance -= price
                self.position = 1

        # 포트폴리오 가치 계산
        self.asset_value = self.balance + self.position * price

        # 보유 비용 적용
        holding_cost = 0
        if self.position == 1:
            holding_cost = 0.001 * price
            self.asset_value -= holding_cost

        # 보상 계산
        reward = (self.asset_value - prev_asset_value) * 10

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # 히스토리 업데이트
        self.balance_history.append(self.balance)
        self.asset_value_history.append(self.asset_value)
        self.position_history.append(self.position)
        self.action_history.append(action)
        self.price_history.append(price)

        obs = self._next_observation()
        return obs, reward, done, {}

# 환경 생성
env = StockTradingEnv(df, scaler)

# 포지셔닝 인코딩 정의
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # 배치 차원을 추가

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]

# 트랜스포머 모델 정의 (포지셔닝 인코딩 포함)
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim=128, n_heads=4, n_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.transformer = nn.Transformer(d_model=model_dim, nhead=n_heads, num_encoder_layers=n_layers)
        self.fc = nn.Linear(model_dim, 3)  # 행동의 확률 분포

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)  # 포지셔닝 인코딩 추가
        x = x.unsqueeze(1)  # 배치 차원 추가
        x = self.transformer(x)
        return torch.softmax(self.fc(x[-1]), dim=-1)  # 마지막 출력을 사용

# PPO를 위한 액터-크리틱 신경망 정의
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.model = TransformerModel(input_dim)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

        # 가중치 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.model(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        policy_logits, _ = self.forward(state)
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob

    def evaluate(self, state, action):
        state = torch.FloatTensor(state)
        policy_logits, value = self.forward(state)
        dist = Categorical(logits=policy_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, value.squeeze(-1), dist_entropy

# 하이퍼파라미터 설정
learning_rate = 1e-3
gamma = 0.99
epsilon = 0.1
epochs = 10
entropy_coef = 0.05

# 정책 및 옵티마이저 초기화
input_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy = ActorCritic(input_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# 메모리 클래스 정의
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

memory = Memory()

# PPO 업데이트 함수 정의
def ppo_update():
    # 리스트를 텐서로 변환
    states = torch.tensor(memory.states, dtype=torch.float32)
    actions = torch.tensor(memory.actions, dtype=torch.int64)
    old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
    rewards = memory.rewards
    is_terminals = memory.is_terminals

    # 리턴 계산
    returns = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        returns.insert(0, discounted_reward)
    returns = torch.tensor(returns, dtype=torch.float32)

    # 어드밴티지 계산
    with torch.no_grad():
        _, state_values = policy.forward(states)
        advantages = returns - state_values.squeeze(-1)

    # 정책 업데이트
    for _ in range(epochs):
        logprobs, state_values, dist_entropy = policy.evaluate(states, actions)
        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
        loss = -torch.min(surr1, surr2) + 0.5 * advantages.pow(2) - entropy_coef * dist_entropy  # 엔트로피 가중치 증가

        optimizer.zero_grad()
        loss.mean().backward()
        # 그레이디언트 클리핑
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        # 손실 값 출력
        print(f"Loss: {loss.mean().item():.4f}")

# 학습 루프
max_episodes = 10  # 에피소드 수를 줄여 빠른 테스트
update_interval = 1

# 행동 분포 추적을 위한 리스트
action_counts = []

for episode in range(max_episodes):
    print(f'episode{episode+1} 시작')
    state = env.reset()
    done = False
    episode_actions = []
    while not done:
        action, action_logprob = policy.act(state)
        next_state, reward, done, _ = env.step(action)
        # 메모리에 데이터 저장
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob.item())
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        state = next_state
        episode_actions.append(action)

    # 정책 업데이트 및 메모리 초기화
    ppo_update()
    memory.clear()

    # 행동 분포 추적
    action_counts.append(np.bincount(episode_actions, minlength=3))

    # 에피소드별 성과 시각화
    plt.figure(figsize=(12, 10))

    # 포트폴리오 가치 변화 시각화
    plt.subplot(5, 1, 1)
    plt.plot(env.asset_value_history)
    plt.title(f'Episode {episode+1} - Asset Value Over Time')
    plt.ylabel('Asset Value')

    # 포지션 변화 시각화
    plt.subplot(5, 1, 2)
    plt.plot(env.position_history)
    plt.title('Position Over Time')
    plt.ylabel('Position')

    # 주가 변화 시각화
    plt.subplot(5, 1, 3)
    plt.plot(env.price_history)
    plt.title('Price Over Time')
    plt.ylabel('Price')

    # 행동 시각화
    plt.subplot(5, 1, 4)
    plt.plot(env.action_history)
    plt.title('Actions Over Time')
    plt.ylabel('Action')
    plt.xlabel('Time Step')
    plt.yticks([0, 1, 2], ['Sell', 'Hold', 'Buy'])

    # 행동 분포 시각화
    plt.subplot(5, 1, 5)
    counts = np.array(action_counts).sum(axis=0)
    plt.bar(['Sell', 'Hold', 'Buy'], counts)
    plt.title('Action Distribution')
    plt.ylabel('Counts')

    plt.tight_layout()
    plt.show()

    print(f"Episode {episode+1} completed. Final Asset Value: {env.asset_value_history[-1]:.2f}")

# 전체 행동 분포 시각화
total_counts = np.array(action_counts).sum(axis=0)
plt.figure(figsize=(6, 4))
plt.bar(['Sell', 'Hold', 'Buy'], total_counts)
plt.title('Total Action Distribution')
plt.ylabel('Counts')
plt.show()

# PPO 알고리즘 적용한 트레이딩 예제