import json
import subprocess

# Charger les emplacements optimisés depuis le fichier JSON
def load_optimized_locations(filename="optimized_locations.json"):
    with open(filename, 'r') as f:
        return json.load(f)

# Utiliser les emplacements optimisés pour configurer les contrôleurs SDN
def deploy_controllers(optimized_locations):
    print("Déploiement des contrôleurs SDN avec les emplacements optimisés...")
    
    for location in optimized_locations:
        controller_id = location["controller_id"]
        switch = location["switch"]
        latency = location["latency"]
        
        # Exemple : Configurer chaque contrôleur sur le switch correspondant
        # Ici, vous pourriez utiliser des outils comme OpenDaylight ou d'autres pour configurer les contrôleurs SDN
        print(f"Déploiement du contrôleur {controller_id} sur le switch {switch} avec une latence de {latency} ms.")
        
        # Simulation de commande pour déployer le contrôleur
        # subprocess.run(["your_deployment_tool", f"--controller_id={controller_id}", f"--switch={switch}", f"--latency={latency}"], check=True)
        
    print("Déploiement terminé.")

# Exemple d'utilisation
optimized_locations = load_optimized_locations()
deploy_controllers(optimized_locations)
