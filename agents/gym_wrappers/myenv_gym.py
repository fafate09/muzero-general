import gym
from gym import spaces
import numpy as np
from games.game1 import Game  # Assurez-vous que le chemin est correct pour Game

# Enregistrer l'environnement MyEnvGym-v0
from gym.envs.registration import register
register(
    id='MyEnvGym-v0',
    entry_point='agents.gym_wrappers.myenv_gym:MyEnvGym',  # Le chemin vers ta classe MyEnvGym
)

class MyEnvGym(gym.Env):
    def __init__(self):
        super(MyEnvGym, self).__init__()
        self.game = Game()  # 👈 Désactive l'affichage pendant l'entraînement
        obs = self.game.reset()
        obs = np.array(obs, dtype=np.float32)

        # Définir l'espace d'action
        self.action_space = spaces.Discrete(len(self.game.legal_actions()))

        # Définir l'espace d'observation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs = self.game.reset()  # Réinitialise l'environnement du jeu
        return np.array(obs, dtype=np.float32), {}  # Retourner l'observation et un dictionnaire vide

    def step(self, action):
        # Assurez-vous que l'action est dans une liste, même si c'est une seule action
        if isinstance(action, int):  # Si l'action est un entier, transforme-la en liste
            action = [action]
        
        # Effectuer le pas dans le jeu
        obs, reward, done = self.game.step(action)
        
        # Convertir l'observation en tableau numpy
        obs = np.array(obs, dtype=np.float32)
        
        # Convertir la récompense en float
        reward = float(reward)
        
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass  # Aucune sortie nécessaire
