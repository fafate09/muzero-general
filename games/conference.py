import datetime
import pathlib
import networkx as nx
import numpy as np
from collections import deque
import numpy
import torch
from top import OS3EWeightedGraph
from .abstract_game import AbstractGame
import time
import os
class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (34, 2, 1)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(101))  # Fixed list of all possible actions. You should only edit the length
        #self.action_space = [0, 1, 2, 3, 4,]        
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 500  # Maximum number of moves if game is not finished before
        self.num_simulations = 100  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 35  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.02  # Initial learning rate
        self.lr_decay_rate = 0.8  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000



        ### Replay Buffer
        self.replay_buffer_size = 500  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1.5  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

import numpy as np
import os
import time
from collections import deque
import networkx as nx
from .abstract_game import AbstractGame  # selon ton arborescence
import networkx as nx
import numpy as np
import os
import time

"""class Game(AbstractGame):
    def __init__(self, seed=None):
        self.graph = OS3EWeightedGraph()
        self.latency_target = 0.9
        self.latency_measure = self.calculate_latency()
        self.winning_controllers = []
        self.max_latency = 0.9
        self.remaining_active_nodes = deque()
        self.current_players = list(range(len(self.graph.nodes())))
        self.reward_curve = []
        self.latency_reward_trace = []  # Liste pour stocker (latency, reward) de chaque action testée


    def step(self, actions):
        start_time = time.time()
        legal_actions = self.legal_actions()

        if not legal_actions:
            raise ValueError("No legal actions available.")

        if not isinstance(actions, list):
            actions = [actions]

        for action in actions:
            if action not in legal_actions:
                raise ValueError(f"Invalid action {action}. Choose from legal actions: {legal_actions}")

        for node in self.graph.nodes:
            node_name = self.graph.nodes[node].get("label", f"Node {node}")
            print(f"(SelfPlay pid=2136) Processing node {node_name}")

            current_status = self.graph.nodes[node].get("active", 0)
            results_for_node = []

            for action in legal_actions:
                print(f"(SelfPlay pid=2136) Trying action {action}")
                self.graph.nodes[node]["active"] = 1 - current_status
                self.graph.nodes[node]["value"] = 1 - current_status
                self.update_state(action)

                new_latency_measure = self.calculate_latency()
                reward_for_action = self.calculate_reward(new_latency_measure)
                print(f"    ➤ Action {action} → latency: {round(new_latency_measure, 4)} | Reward: {round(reward_for_action, 4)}")
                self.latency_reward_trace.append((new_latency_measure, reward_for_action))

                results_for_node.append((action, new_latency_measure))

                self.graph.nodes[node]["active"] = current_status
                self.graph.nodes[node]["value"] = current_status

            print(f"(SelfPlay pid=2136) Results for node {node_name}:")
            for action_result in sorted(results_for_node, key=lambda x: -x[1]):
                print(f"    (Action: {action_result[0]}, Score: {round(action_result[1], 4)})")

            best_action_for_node = max(results_for_node, key=lambda x: x[1])
            print(f"(SelfPlay pid=2136) Best action for node {node_name}: {best_action_for_node}")

            self.graph.nodes[node]["active"] = 1 - current_status
            self.graph.nodes[node]["value"] = 1 - current_status
            self.update_state(best_action_for_node[0])

        final_latency_measure = self.calculate_latency()
        reward = self.calculate_reward(final_latency_measure)
        self.reward_curve.append(reward)
        done = self.is_winner()

        winning_controllers = [node for node in self.graph.nodes if self.is_node_active(node)]
        node_labels = [self.graph.nodes[n].get("label", f"Node {n}") for n in winning_controllers]

        print("=" * 60)
        print(f"(SelfPlay pid=2136) FINAL SUMMARY")
        print(f"(SelfPlay pid=2136) Optimal Nodes: {node_labels}")
        print(f"(SelfPlay pid=2136) Number of Optimal Nodes: {len(winning_controllers)}")
        print(f"(SelfPlay pid=2136) Final latency: {round(final_latency_measure, 4)}")
        print(f"(SelfPlay pid=2136) Played action: {actions[0]}")
        print("=" * 60)

        print(f"(SelfPlay) Étape {len(self.reward_curve)} - Récompense : {reward}")

        return self.get_observation(), reward, done

    def save_results(self, runtime):
        results_muzero = {
            "final_latency": self.calculate_latency(),
            "reward": self.reward_curve[-1] if self.reward_curve else 0,
            "num_nodes": len(self.get_active_nodes()),
            "evals": len(self.graph.nodes()) ** 2,
            "runtime": runtime,
            "reward_curve": self.reward_curve,
            "latency_reward_trace": self.latency_reward_trace,

        }

        if os.path.exists("results_muzero.npy"):
            previous = np.load("results_muzero.npy", allow_pickle=True).tolist()
            if not isinstance(previous, list):
                previous = [previous]
        else:
            previous = []

        previous.append(results_muzero)
        np.save("results_muzero.npy", previous)
        print("✅ Courbe complète de récompenses sauvegardée dans 'results_muzero.npy'")

    # Le reste des méthodes reste inchangé :
    def reset(self):
        self.latency_measure = self.calculate_latency()
        for node in self.graph.nodes:
            self.graph.nodes[node]["value"] = 0
            self.graph.nodes[node]["active"] = 1
        self.remaining_active_nodes = deque(self.get_active_nodes())
        self.reward_curve = []
        self.latency_reward_trace = []
        return self.get_observation()

    def calculate_latency(self):
        active_nodes = self.get_active_nodes()
        latency_score = 0
        for node1 in active_nodes:
            for node2 in active_nodes:
                if node1 != node2:
                    latency_score += self.calculate_latency_between_nodes(node1, node2)
        return latency_score

    def calculate_latency_between_nodes(self, node1, node2):
        METERS_TO_MILES = 1609.34
        SPEED_OF_LIGHT = 3 * 1e8
        if not nx.has_path(self.graph, node1, node2):
            return 0
        shortest_path_length = nx.shortest_path_length(self.graph, node1, node2, weight='weight')
        latency = shortest_path_length / METERS_TO_MILES / SPEED_OF_LIGHT * 1000
        return 1 / latency

    def calculate_reward(self, new_latency_measure):
        latency_difference = self.latency_measure - new_latency_measure
        reward_scale = 1e6
        return abs(latency_difference / reward_scale)

    def update_state(self, action):
        node = list(self.graph.nodes())[action]
        current_status = self.graph.nodes[node].get("active", 0)
        self.graph.nodes[node]["active"] = 1 - current_status
        self.graph.nodes[node]["value"] = self.current_players[0]

    def is_winner(self):
        current_latency = self.latency_measure
        max_latency = current_latency
        best_configuration = None

        for node in self.graph.nodes:
            current_status = self.graph.nodes[node].get("active", 0)
            self.graph.nodes[node]["active"] = 1 - current_status
            self.graph.nodes[node]["value"] = 1 - current_status
            new_latency_measure = self.calculate_latency()

            if new_latency_measure > max_latency:
                max_latency = new_latency_measure
                best_configuration = dict(self.graph.nodes(data=True))

            self.graph.nodes[node]["active"] = current_status
            self.graph.nodes[node]["value"] = current_status
            self.latency_measure = current_latency

        if best_configuration:
            self.winning_controllers = [node for node, data in best_configuration.items() if data.get("active", 0) == 1]
            print("Optimal Number of Nodes:", len(self.winning_controllers))

        return best_configuration is not None

    def is_node_active(self, node):
        if node not in self.graph.nodes:
            raise ValueError(f"Node {node} not found in the graph.")
        return self.graph.nodes[node].get("active", 0) == 1

    def get_active_nodes(self):
        return [node for node in self.graph.nodes if self.is_node_active(node)]

    def legal_actions(self):
        return list(range(len(self.graph.nodes)))

    def render(self):
        print("Active Nodes:", self.legal_actions())

    def get_observation(self):
        nodes_list = list(self.graph.nodes())
        node_to_index = {node: index for index, node in enumerate(nodes_list)}
        observation = np.array([[float(node_to_index[node]), float(self.latency_measure)] for node in nodes_list])
        observation = observation[:, :, np.newaxis]
        return observation
"""
class Game(AbstractGame):
    def __init__(self, seed=None):
        self.graph = OS3EWeightedGraph()
        self.latency_target = 0.9
        self.latency_measure = self.calculate_latency()
        self.winning_controllers = []
        self.max_latency = 0.9
        self.remaining_active_nodes = deque()
        self.current_players = list(range(len(self.graph.nodes())))
        self.reward_curve = []
        self.latency_reward_trace = []  # Liste pour stocker (latency, reward) de chaque action testée


    def step(self, actions):
        start_time = time.time()
        legal_actions = self.legal_actions()

        if not legal_actions:
            raise ValueError("No legal actions available.")

        if not isinstance(actions, list):
            actions = [actions]

        for action in actions:
            if action not in legal_actions:
                raise ValueError(f"Invalid action {action}. Choose from legal actions: {legal_actions}")

        for node in self.graph.nodes:
            node_name = self.graph.nodes[node].get("label", f"Node {node}")
            print(f"(SelfPlay pid=2136) Processing node {node_name}")

            current_status = self.graph.nodes[node].get("active", 0)
            results_for_node = []

            for action in legal_actions:
                print(f"(SelfPlay pid=2136) Trying action {action}")
                self.graph.nodes[node]["active"] = 1 - current_status
                self.graph.nodes[node]["value"] = 1 - current_status
                self.update_state(action)

                new_latency_measure = self.calculate_latency()  # latency now reflects latency
                reward_for_action = self.calculate_reward(new_latency_measure)
                print(f"    ➤ Action {action} → latency (Latence): {round(new_latency_measure, 4)} | Reward: {round(reward_for_action, 4)}")
                self.latency_reward_trace.append((new_latency_measure, reward_for_action))

                results_for_node.append((action, new_latency_measure))

                self.graph.nodes[node]["active"] = current_status
                self.graph.nodes[node]["value"] = current_status

            print(f"(SelfPlay pid=2136) Results for node {node_name}:")
            for action_result in sorted(results_for_node, key=lambda x: -x[1]):
                print(f"    (Action: {action_result[0]}, Score: {round(action_result[1], 4)})")

            best_action_for_node = max(results_for_node, key=lambda x: x[1])
            print(f"(SelfPlay pid=2136) Best action for node {node_name}: {best_action_for_node}")

            self.graph.nodes[node]["active"] = 1 - current_status
            self.graph.nodes[node]["value"] = 1 - current_status
            self.update_state(best_action_for_node[0])

        final_latency_measure = self.calculate_latency()
        reward = self.calculate_reward(final_latency_measure)
        self.reward_curve.append(reward)
        done = self.is_winner()

        winning_controllers = [node for node in self.graph.nodes if self.is_node_active(node)]
        node_labels = [self.graph.nodes[n].get("label", f"Node {n}") for n in winning_controllers]

        print("=" * 60)
        print(f"(SelfPlay pid=2136) FINAL SUMMARY")
        print(f"(SelfPlay pid=2136) Optimal Nodes: {node_labels}")
        print(f"(SelfPlay pid=2136) Number of Optimal Nodes: {len(winning_controllers)}")
        print(f"(SelfPlay pid=2136) Final latency (Latence): {round(final_latency_measure, 4)}")
        print(f"(SelfPlay pid=2136) Played action: {actions[0]}")
        print("=" * 60)

        print(f"(SelfPlay) Étape {len(self.reward_curve)} - Récompense : {reward}")

        return self.get_observation(), reward, done

    def save_results(self, runtime):
        results_muzero = {
            "final_latency": self.calculate_latency(),
            "reward": self.reward_curve[-1] if self.reward_curve else 0,
            "num_nodes": len(self.get_active_nodes()),
            "evals": len(self.graph.nodes()) ** 2,
            "runtime": runtime,
            "reward_curve": self.reward_curve,
            "latency_reward_trace": self.latency_reward_trace,

        }

        if os.path.exists("results_muzero.npy"):
            previous = np.load("results_muzero.npy", allow_pickle=True).tolist()
            if not isinstance(previous, list):
                previous = [previous]
        else:
            previous = []

        previous.append(results_muzero)
        np.save("results_muzero.npy", previous)
        print("✅ Courbe complète de récompenses sauvegardée dans 'results_muzero.npy'")

    # Le reste des méthodes reste inchangé :
    def reset(self):
        self.latency_measure = self.calculate_latency()
        for node in self.graph.nodes:
            self.graph.nodes[node]["value"] = 0
            self.graph.nodes[node]["active"] = 1
        self.remaining_active_nodes = deque(self.get_active_nodes())
        self.reward_curve = []
        self.latency_reward_trace = []
        return self.get_observation()

    def calculate_latency(self):
        # Initialisation du score de cohérence basé sur la latence
        active_nodes = self.get_active_nodes()
        total_latency = 0
        for node1 in active_nodes:
            for node2 in active_nodes:
                if node1 != node2:
                    total_latency += self.calculate_latency_between_nodes(node1, node2)
        # Réduire le score de cohérence en fonction de la latence totale
        return total_latency

    def calculate_latency_between_nodes(self, node1, node2):
        METERS_TO_MILES = 1609.34
        SPEED_OF_LIGHT = 3 * 1e8  # en m/s
        if not nx.has_path(self.graph, node1, node2):
            return float('inf')  # Si aucun chemin n'existe, la latence est infinie
        shortest_path_length = nx.shortest_path_length(self.graph, node1, node2, weight='weight')
        latency = shortest_path_length / METERS_TO_MILES / SPEED_OF_LIGHT * 1000  # latence en ms
        return latency

    def calculate_reward(self, new_latency_measure):
        # Calcul de la différence de latence
        latency_difference = self.latency_measure - new_latency_measure
        print(f"latency_difference: {latency_difference}")  # Vérifier la différence

        #reward_scale = 1e3  # Ajustez le facteur d'échelle selon les résultats

        # Calcul de la récompense
        #reward = max(0, latency_difference / reward_scale)
        reward=10
        reward*=2
        print(f"Calculated reward: {reward}")  # Vérifier la récompense

        return reward

    def update_state(self, action):
        node = list(self.graph.nodes())[action]
        current_status = self.graph.nodes[node].get("active", 0)
        self.graph.nodes[node]["active"] = 1 - current_status
        self.graph.nodes[node]["value"] = self.current_players[0]

    def is_winner(self):
        current_latency = self.latency_measure
        max_latency = current_latency
        best_configuration = None

        for node in self.graph.nodes:
            current_status = self.graph.nodes[node].get("active", 0)
            self.graph.nodes[node]["active"] = 1 - current_status
            self.graph.nodes[node]["value"] = 1 - current_status
            new_latency_measure = self.calculate_latency()

            if new_latency_measure > max_latency:
                max_latency = new_latency_measure
                best_configuration = dict(self.graph.nodes(data=True))

            self.graph.nodes[node]["active"] = current_status
            self.graph.nodes[node]["value"] = current_status
            self.latency_measure = current_latency

        if best_configuration:
            self.winning_controllers = [node for node, data in best_configuration.items() if data.get("active", 0) == 1]
            print("Optimal Number of Nodes:", len(self.winning_controllers))

        return best_configuration is not None

    def is_node_active(self, node):
        if node not in self.graph.nodes:
            raise ValueError(f"Node {node} not found in the graph.")
        return self.graph.nodes[node].get("active", 0) == 1

    def get_active_nodes(self):
        return [node for node in self.graph.nodes if self.is_node_active(node)]

    def legal_actions(self):
        return list(range(len(self.graph.nodes)))

    def render(self):
        print("Active Nodes:", self.legal_actions())

    def get_observation(self):
        nodes_list = list(self.graph.nodes())
        node_to_index = {node: index for index, node in enumerate(nodes_list)}
        observation = np.array([[float(node_to_index[node]), float(self.latency_measure)] for node in nodes_list])
        observation = observation[:, :, np.newaxis]
        return observation

