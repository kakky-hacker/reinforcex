import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import requests
import time

# -------- Q-Network --------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
        )

    def forward(self, x):
        return self.model(x)

# -------- Replay Buffer --------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.index = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = transition
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -------- DQN Trainer --------
def train_dqn_api(env_url: str, episodes=1000):
    state_dim = 8
    action_dim = 4

    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=3e-4)
    replay_buffer = ReplayBuffer(4000)

    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 10000
    gamma = 0.99
    batch_size = 64
    steps = 0

    def epsilon_by_step(step):
        return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay)
    stime=time.time()
    for episode in range(episodes):
        # --- POST /reset ---
        while True:
            try:
                reset_resp = requests.post(f"{env_url}/reset", json={"env": "LunarLander-v3"}).json()
                break
            except:
                continue
        session_id = reset_resp["session_id"]
        state = np.array(reset_resp["observation"], dtype=np.float32)

        total_reward = 0.0
        for t in range(1000):
            epsilon = epsilon_by_step(steps)
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = q_net(state_tensor).argmax().item()

            # --- POST /step ---
            try:
                step_resp = requests.post(f"{env_url}/step", json={"session_id": session_id, "action": action}).json()
            except:
                pass
            next_state = np.array(step_resp["observation"], dtype=np.float32)
            reward = step_resp["reward"]
            done = step_resp["done"]

            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1

            # 学習
            if len(replay_buffer) >= batch_size and steps % 8 == 0:
                s, a, r, s2, d = replay_buffer.sample(batch_size)
                s = torch.FloatTensor(s)
                a = torch.LongTensor(a).unsqueeze(1)
                r = torch.FloatTensor(r)
                s2 = torch.FloatTensor(s2)
                d = torch.FloatTensor(d)

                q = q_net(s).gather(1, a).squeeze()
                with torch.no_grad():
                    q_next = target_net(s2).max(1)[0]
                    target = r + gamma * q_next * (1 - d)

                loss = nn.MSELoss()(q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # ターゲットネット更新
        if episode % 50 == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {episode}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.2f}, time: {time.time()-stime}")

# -------- 実行例 --------
if __name__ == "__main__":
    train_dqn_api("http://localhost:8001")
