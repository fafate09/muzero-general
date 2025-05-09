import gym
import numpy as np
import os
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 🔧 Pour que l'import du jeu fonctionne
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_wrappers.myenv_gym import MyEnvGym  # Assurez-vous que le chemin est correct pour MyEnvGym

# 📦 Debug dans l’environnement
class DebugEnv(MyEnvGym):
    def reset(self, **kwargs):
        print(">>> RESET called")
        obs = super().reset()
        print(">>> Initial observation:", obs)
        return obs

    def step(self, action):
        print(f">>> STEP called with action: {action}")

        # Assurer que l'action est dans un tableau, même si c'est un seul entier
        if isinstance(action, int):
            action = [action]  # Transforme en liste si c'est un seul entier

        # Appeler la méthode step de la classe parente
        obs, reward, done, info = super().step(action)
        
        print(f"    -> Reward: {reward}, Done: {done}")
        
        return obs, reward, done, info

# 🔁 Créer et envelopper l'environnement
def make_env():
    env = DebugEnv()
    env = Monitor(env)  # Log progression
    return env

env = DummyVecEnv([make_env])  # Assurez-vous d'envelopper correctement l'environnement

# ✅ Test rapide du fonctionnement de l’env
print("\n=== TEST ENVIRONNEMENT ===")
obs = env.reset()
for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print(f"[Test] Step {i}: Action={action}, Reward={reward}, Done={done}")
    if done:
        print("[Test] Episode terminé")
        break

# 📊 Entraînement SB3
print("\n=== ENTRAÎNEMENT DQN ===")
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/sb3_dqn"
)

model.learn(total_timesteps=10000, tb_log_name="debug_run")
print("✅ Apprentissage terminé.")
