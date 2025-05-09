# dqn_game1.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import random
from collections import deque, namedtuple
import gym
from gym import spaces
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from top import OS3EWeightedGraph

# ----------- ENVIRONNEMENT -----------

class DQNEnergyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.graph = OS3EWeightedGraph()
        for node in self.graph.nodes():
            self.graph.nodes[node]["active"] = 1
            self.graph.nodes[node]["value"] = 0

        self.coherence_measure = self.calculate_coherence()
        self.action_space = spaces.Discrete(len(self.graph.nodes))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.graph.nodes), 2), dtype=np.float32
        )
        self.current_step = 0
        self.max_steps = 20

    def calculate_coherence_between_nodes(self, node1, node2):
        METERS_TO_MILES = 1609.34
        SPEED_OF_LIGHT = 3e8
        if not nx.has_path(self.graph, node1, node2):
            return 0
        shortest_path = nx.shortest_path_length(self.graph, node1, node2, weight="weight")
        latency = shortest_path / METERS_TO_MILES / SPEED_OF_LIGHT * 1000
        return 1 / latency

    def calculate_coherence(self):
        active_nodes = [n for n in self.graph.nodes if self.graph.nodes[n]["active"]]
        coherence = 0
        for i in active_nodes:
            for j in active_nodes:
                if i != j:
                    coherence += self.calculate_coherence_between_nodes(i, j)
        return coherence

    def calculate_reward(self, new_measure):
        diff = abs(new_measure - self.coherence_measure)
        return diff / 1e6

    def get_observation(self):
        obs = []
        for n in self.graph.nodes:
            obs.append([float(self.graph.nodes[n]["active"]), float(self.coherence_measure)])
        return np.array(obs, dtype=np.float32)

    def reset(self):
        for n in self.graph.nodes:
            self.graph.nodes[n]["active"] = 1
            self.graph.nodes[n]["value"] = 0
        self.coherence_measure = self.calculate_coherence()
        self.current_step = 0
        return self.get_observation()

    def step(self, action):
        self.current_step += 1
        node = list(self.graph.nodes)[action]
        current_status = self.graph.nodes[node]["active"]
        self.graph.nodes[node]["active"] = 1 - current_status
        self.graph.nodes[node]["value"] = 1 - current_status
        new_coherence = self.calculate_coherence()
        reward = self.calculate_reward(new_coherence)
        self.coherence_measure = new_coherence
        done = self.current_step >= self.max_steps
        return self.get_observation(), reward, done, {}

# ----------- DQN AGENT -----------

class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity=500):  # MuZero uses 500
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ----------- TRAINING LOOP -----------

def train_dqn():
    env = DQNEnergyEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    policy_net = QNetwork(input_dim, action_dim).to(device)
    target_net = QNetwork(input_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.02)
    buffer = ReplayBuffer()
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    batch_size = 128
    gamma = 0.997
    update_target_every = 10  # update freq

    writer = SummaryWriter("runs/dqn_energy")
    reward_history = []
    coherence_history = []

    for episode in range(200):
        state = env.reset().flatten()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q = policy_net(s)
                    action = q.argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                transitions = buffer.sample(batch_size)
                batch = Transition(*zip(*transitions))

                state_batch = torch.FloatTensor(batch.state).to(device)
                action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
                reward_batch = torch.FloatTensor(batch.reward).to(device)
                next_state_batch = torch.FloatTensor(batch.next_state).to(device)
                done_batch = torch.FloatTensor(batch.done).to(device)

                q_values = policy_net(state_batch).gather(1, action_batch).squeeze()
                next_q = target_net(next_state_batch).max(1)[0]
                target = reward_batch + gamma * next_q * (1 - done_batch)

                loss = F.mse_loss(q_values, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reward_history.append(total_reward)
        coherence_history.append(env.coherence_measure)

        writer.add_scalar("Reward/Total", total_reward, episode)
        writer.add_scalar("Coherence/Final", env.coherence_measure, episode)

        print(f"Episode {episode}, Total reward: {total_reward:.4f}, Epsilon: {epsilon:.3f}")

    # Plotting
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(reward_history)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(coherence_history)
    plt.title("Final Coherence per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Coherence")

    plt.tight_layout()
    plt.savefig("dqn_training_curves.png")
    plt.show()

# ----------- MAIN -----------

if __name__ == "__main__":
    train_dqn()
