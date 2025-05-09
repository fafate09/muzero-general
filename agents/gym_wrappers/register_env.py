# gym_wrappers/register_env.py
from gym.envs.registration import register

# Enregistrer l'environnement MyEnvGym
register(
    id='MyEnvGym-v0',  # ID de l'environnement
    entry_point='gym_wrappers.myenv_gym:MyEnvGym',  # Chemin vers la classe MyEnvGym
)
