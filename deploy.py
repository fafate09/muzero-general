import json
import subprocess

# Charger les emplacements optimisés depuis le fichier JSON
def load_optimized_locations(filename="optimized_locations.json"):
    with open(filename, 'r') as f:
        return json.load(f)

# Utiliser les emplacements optimisés pour configurer les contrôleurs SDN
def deploy_controllers(optimized_locations):
    print("Déploiement des contrôleurs SDN avec les emplacements optimisés...")

    for i, switch in enumerate(optimized_locations):
        controller_id = f"ctrl-{i}"
        latency = 10  # Tu peux ajuster ou calculer une latence si nécessaire

        print(f"Déploiement du contrôleur {controller_id} sur le switch {switch} avec une latence de {latency} ms.")

        # Simulation de commande pour déployer le contrôleur
        # subprocess.run(["your_deployment_tool", f"--controller_id={controller_id}", f"--switch={switch}", f"--latency={latency}"], check=True)

    print("Déploiement terminé.")

# Exemple d'utilisation
optimized_locations = load_optimized_locations()
deploy_controllers(optimized_locations)

