import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import time

# Définition du modèle DQN
class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(observation_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ReplayBuffer pour stocker les transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Agent DQN
class DQNAgent:
    def __init__(self, observation_space, action_space, batch_size=32, gamma=0.99, epsilon=0.1, lr=0.001, buffer_size=10000):
        self.action_space = action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.memory = ReplayBuffer(buffer_size)
        
        self.model = DQN(observation_space, action_space)
        self.target_model = DQN(observation_space, action_space)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_space))
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def update(self):
        if self.memory.size() < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        target = rewards + (self.gamma * next_q_value * (1 - dones))

        loss = self.loss_fn(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Fonction pour entraîner l'agent DQN
def train_dqn(agent, game, num_episodes=1000, max_timesteps=1000):
    rewards_per_episode = []
    total_steps = 0
    start_time = time.time()

    for episode in range(num_episodes):
        state = game.reset()  # Assurez-vous d'avoir une méthode `reset()` qui renvoie l'état initial
        episode_reward = 0

        for t in range(max_timesteps):
            action = agent.select_action(state)
            next_state, reward, done = game.step(action)  # Assurez-vous que `step()` renvoie (state, reward, done)
            agent.memory.add(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                break

        rewards_per_episode.append(episode_reward)

        # Mise à jour du modèle cible tous les 10 épisodes
        if episode % 10 == 0:
            agent.update_target()

        # Affichage des métriques
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} | Reward: {episode_reward} | Time Elapsed: {round(time.time() - start_time, 2)}s")

    return rewards_per_episode

# Fonction pour afficher les résultats
def plot_rewards(rewards_per_episode):
    plt.plot(rewards_per_episode)
    plt.xlabel("Épisodes")
    plt.ylabel("Récompense cumulée")
    plt.title("Performance de l'agent DQN")
    plt.show()

# Entraînement de l'agent DQN
if __name__ == "__main__":
    # Crée l'environnement (Game)
    game = Game()  # Assurez-vous que vous avez la classe `Game` définie et qu'elle fonctionne correctement

    # Crée un agent DQN
    agent_dqn = DQNAgent(observation_space=34, action_space=101)  # Par exemple, 34 dimensions pour l'état et 101 actions possibles
    rewards_per_episode = train_dqn(agent_dqn, game, num_episodes=1000, max_timesteps=1000)

    # Afficher les courbes de récompenses
    plot_rewards(rewards_per_episode)
