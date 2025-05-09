import gym
from stable_baselines3 import DQN
from stable_baselines3 import MuZero
from game1 import Game1Env  # Ton environnement personnalisé

# Créer l'environnement personnalisé
env = Game1Env()

# Entraîner l'agent DQN
print("Entraînement de DQN...")
dqn_model = DQN("MlpPolicy", env, verbose=1)
dqn_model.learn(total_timesteps=10000)

# Sauvegarder le modèle DQN
dqn_model.save("dqn_game1")

# Tester l'agent DQN
obs = env.reset()
done = False
total_reward_dqn = 0

while not done:
    action, _states = dqn_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward_dqn += reward
    if done:
        print(f"Récompense totale DQN: {total_reward_dqn}")
        obs = env.reset()


# Entraîner l'agent MuZero
print("Entraînement de MuZero...")
muzero_model = MuZero("MlpPolicy", env, verbose=1)
muzero_model.learn(total_timesteps=10000)

# Sauvegarder le modèle MuZero
muzero_model.save("muzero_game1")

# Tester l'agent MuZero
obs = env.reset()
done = False
total_reward_muzero = 0

while not done:
    action, _states = muzero_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward_muzero += reward
    if done:
        print(f"Récompense totale MuZero: {total_reward_muzero}")
        obs = env.reset()
