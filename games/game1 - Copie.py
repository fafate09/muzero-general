import datetime
import pathlib
import networkx as nx
import numpy as np
from collections import deque
import numpy
import torch
from top import OS3EWeightedGraph
import datetime
import pathlib
import networkx as nx
import numpy as np
from collections import deque
import numpy
import torch
from top import OS3EWeightedGraph
from .abstract_game import AbstractGame

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (34, 2, 1)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(34))  # Fixed list of all possible actions. You should only edit the length
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
        self.num_simulations = 20  # Number of future moves self-simulated
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
        self.encoding_size = 64
        self.fc_representation_layers = [64,64]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [64]  # Define the hidden layers in the value network
        self.fc_policy_layers = [64]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 300  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.001  # Initial learning rate
        self.lr_decay_rate = 0.8  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 5000



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
class Game(AbstractGame):
    def __init__(self, seed=None):
        self.graph = OS3EWeightedGraph()
        self.coherence_target = 0.9
        self.coherence_measure = None
        self.winning_controllers = []
        self.max_coherence = 0.9
        self.remaining_active_nodes = deque()
        self.current_players = list(range(len(self.graph.nodes())))  # Liste des joueurs actuels

    """def step(self, actions):
        legal_actions = self.legal_actions()

        if not legal_actions:
            raise ValueError("No legal actions available.")

        # Assurez-vous que actions est une liste
        if not isinstance(actions, list):
            actions = [actions]

        for action in actions:
            if action not in legal_actions:
                raise ValueError(f"Invalid action {action}. Choose from legal actions: {legal_actions}")

        # Iterez sur tous les noeuds et testez chaque action
        for node in self.graph.nodes:
            for action in legal_actions:
                print(f"Testing action {action} at node {node}:")

                # Toggle the status of the current node
                current_status = self.graph.nodes[node].get("active", 0)
                self.graph.nodes[node]["active"] = 1 - current_status
                self.graph.nodes[node]["value"] = 1 - current_status

                # Update the state based on the action
                self.update_state(action)

                # Print the current state after the update
                self.render()

                # Passer au joueur suivant après chaque action
                self.current_players = [1 - player for player in self.current_players]

                new_coherence_measure = self.calculate_coherence()
                reward = self.calculate_reward(new_coherence_measure)
                done = self.is_winner()

                print(f"Reward: {reward}")
                print(f"New Coherence Measure: {new_coherence_measure}")

                if done:
                    print("Winners:", self.winning_controllers)
                    print("Optimal Number of Nodes:", len(self.winning_controllers))

                # Reset the configuration to its original state
                self.graph.nodes[node]["active"] = current_status
                self.graph.nodes[node]["value"] = current_status

        print("Done")

        return self.get_observation(), reward, done"""

    def step(self, actions):
        legal_actions = self.legal_actions()

        if not legal_actions:
            raise ValueError("No legal actions available.")

        # Assurez-vous que actions est une liste
        if not isinstance(actions, list):
            actions = [actions]

        for action in actions:
            if action not in legal_actions:
                raise ValueError(f"Invalid action {action}. Choose from legal actions: {legal_actions}")
        # Iterez sur tous les noeuds
        for node in self.graph.nodes:
            # Sauvegardez l'état initial du nœud
            current_status = self.graph.nodes[node].get("active", 0)

            # Liste pour stocker les résultats de chaque action pour ce nœud
            results_for_node = []

            # Iterez sur toutes les actions possibles pour ce nœud
            for action in legal_actions:
                # Toggle the status of the current node
                self.graph.nodes[node]["active"] = 1 - current_status
                self.graph.nodes[node]["value"] = 1 - current_status

                # Update the state based on the action
                self.update_state(action)

                # Calculate coherence after all actions for this node
                new_coherence_measure = self.calculate_coherence()

                # Save the results for this action
                results_for_node.append((action, new_coherence_measure))

                # Reset the configuration to its original state
                self.graph.nodes[node]["active"] = current_status
                self.graph.nodes[node]["value"] = current_status

                # Find the action that maximizes coherence for this node
                best_action_for_node = max(results_for_node, key=lambda x: x[1])

                # Toggle the status of the current node based on the best action
                self.graph.nodes[node]["active"] = 1 - current_status
                self.graph.nodes[node]["value"] = 1 - current_status
                self.update_state(best_action_for_node[0])  # Update the state with the best action

                # Calculate coherence after all actions for all nodes
                final_coherence_measure = self.calculate_coherence()
                reward = self.calculate_reward(new_coherence_measure)
                done = self.is_winner()

                # Determine the winning controllers based on active status
                winning_controllers = [node for node in self.graph.nodes if self.is_node_active(node)]

                #print("Winning Configuration:", dict(self.graph.nodes(data=True)))
                print("Optimal Number of Nodes:", len(winning_controllers))
                print("Final Coherence Measure:", final_coherence_measure)
                print("Winners:", self.winning_controllers)
                print("reward",reward)

        return self.get_observation(), reward, done



    def reset(self):
        # Reset the game state
        self.coherence_measure = self.calculate_coherence()

        for node in self.graph.nodes:
            self.graph.nodes[node]["value"] = 0

        # Set the active status of all nodes in the graph during reset
        for node in self.graph.nodes:
            self.graph.nodes[node]["active"] = 1

        self.remaining_active_nodes = deque(self.get_active_nodes())

        return self.get_observation()

    def calculate_coherence(self):
        active_nodes = self.get_active_nodes()
        coherence_score = 0

        for node1 in active_nodes:
            for node2 in active_nodes:
                if node1 != node2:
                    coherence_score += self.calculate_coherence_between_nodes(node1, node2)

        return coherence_score

    def calculate_coherence_between_nodes(self, node1, node2):
        METERS_TO_MILES = 1609.34  # Conversion factor from meters to miles
        SPEED_OF_LIGHT = 3 * 1e8  # Speed of light in meters per second

        if not nx.has_path(self.graph, node1, node2):
            return 0  # No direct connection, coherence is 0

        shortest_path_length = nx.shortest_path_length(self.graph, node1, node2, weight='weight')
        latency = shortest_path_length / METERS_TO_MILES / SPEED_OF_LIGHT * 1000

        # You can add more sophisticated logic here to determine coherence based on latency

        return 1 / latency  # Simple example: higher coherence for lower latency

    """def calculate_reward(self, new_coherence_measure):
        coherence_difference = self.coherence_measure - new_coherence_measure
        return coherence_difference"""
    def calculate_reward(self, new_coherence_measure):
        coherence_difference = self.coherence_measure - new_coherence_measure
        reward_scale = 1e6  # Choisissez une valeur appropriée pour l'échelle de récompense
        reward = abs(coherence_difference / reward_scale)
        return reward

    def update_state(self, action):
        # Met à jour l'état en fonction de l'action
        node = list(self.graph.nodes())[action]
        current_status = self.graph.nodes[node].get("active", 0)
        self.graph.nodes[node]["active"] = 1 - current_status
        self.graph.nodes[node]["value"] = self.current_players[0]  # Met à jour avec l'état actuel du joueur

    def is_winner(self):
        current_coherence = self.coherence_measure

        max_coherence = current_coherence
        best_configuration = None

        for node in self.graph.nodes:
            # Toggle the status of the current node
            current_status = self.graph.nodes[node].get("active", 0)
            self.graph.nodes[node]["active"] = 1 - current_status
            self.graph.nodes[node]["value"] = 1 - current_status

            new_coherence_measure = self.calculate_coherence()

            if new_coherence_measure > max_coherence:
                max_coherence = new_coherence_measure
                best_configuration = dict(self.graph.nodes(data=True))

            # Reset the configuration to its original state
            self.graph.nodes[node]["active"] = current_status
            self.graph.nodes[node]["value"] = current_status

            # Réinitialiser la cohérence à la valeur initiale
            self.coherence_measure = current_coherence

        if best_configuration:
            # Update the winning controllers based on active status
            winning_controllers = [node for node, data in best_configuration.items() if data.get("active", 0) == 1]
            self.winning_controllers = winning_controllers

            #print("Winning Configuration:", best_configuration)
            print("Optimal Number of Nodes:", len(winning_controllers))

        return best_configuration is not None

    def is_node_active(self, node):
        if node not in self.graph.nodes:
            raise ValueError(f"Node {node} not found in the graph.")

        is_active = self.graph.nodes[node].get("active", 0)
        return is_active == 1

    def get_active_nodes(self):
        return [node for node in self.graph.nodes if self.is_node_active(node)]

    def legal_actions(self):
        return list(range(len(self.graph.nodes)))

    def render(self):
        #print("Nodes:", list(self.graph.nodes()))
        #print("Coherence Measure:", self.coherence_measure)
        print("Active Nodes:", self.legal_actions())

    def get_observation(self):
        nodes_list = list(self.graph.nodes())
        node_to_index = {node: index for index, node in enumerate(nodes_list)}
        observation = np.array([[float(node_to_index[node]), float(self.coherence_measure)] for node in nodes_list])
        observation = observation[:, :, np.newaxis]

        return observation

class Game(AbstractGame):
    def __init__(self, seed=None):
        self.graph = OS3EWeightedGraph()
        self.coherence_target = 0.9
        self.coherence_measure = None
        self.winning_controllers = []
        self.max_coherence = 0.9
        self.remaining_active_nodes = deque()
        self.current_players = list(range(len(self.graph.nodes())))  # Liste des joueurs actuels

    """def step(self, actions):
        legal_actions = self.legal_actions()

        if not legal_actions:
            raise ValueError("No legal actions available.")

        # Assurez-vous que actions est une liste
        if not isinstance(actions, list):
            actions = [actions]

        for action in actions:
            if action not in legal_actions:
                raise ValueError(f"Invalid action {action}. Choose from legal actions: {legal_actions}")

        # Iterez sur tous les noeuds et testez chaque action
        for node in self.graph.nodes:
            for action in legal_actions:
                print(f"Testing action {action} at node {node}:")

                # Toggle the status of the current node
                current_status = self.graph.nodes[node].get("active", 0)
                self.graph.nodes[node]["active"] = 1 - current_status
                self.graph.nodes[node]["value"] = 1 - current_status

                # Update the state based on the action
                self.update_state(action)

                # Print the current state after the update
                self.render()

                # Passer au joueur suivant après chaque action
                self.current_players = [1 - player for player in self.current_players]

                new_coherence_measure = self.calculate_coherence()
                reward = self.calculate_reward(new_coherence_measure)
                done = self.is_winner()

                print(f"Reward: {reward}")
                print(f"New Coherence Measure: {new_coherence_measure}")

                if done:
                    print("Winners:", self.winning_controllers)
                    print("Optimal Number of Nodes:", len(self.winning_controllers))

                # Reset the configuration to its original state
                self.graph.nodes[node]["active"] = current_status
                self.graph.nodes[node]["value"] = current_status

        print("Done")

        return self.get_observation(), reward, done"""

    def step(self, actions):
        legal_actions = self.legal_actions()

        if not legal_actions:
            raise ValueError("No legal actions available.")

        # Assurez-vous que actions est une liste
        if not isinstance(actions, list):
            actions = [actions]

        for action in actions:
            if action not in legal_actions:
                raise ValueError(f"Invalid action {action}. Choose from legal actions: {legal_actions}")
        # Iterez sur tous les noeuds
        for node in self.graph.nodes:
            # Sauvegardez l'état initial du nœud
            current_status = self.graph.nodes[node].get("active", 0)

            # Liste pour stocker les résultats de chaque action pour ce nœud
            results_for_node = []

            # Iterez sur toutes les actions possibles pour ce nœud
            for action in legal_actions:
                # Toggle the status of the current node
                self.graph.nodes[node]["active"] = 1 - current_status
                self.graph.nodes[node]["value"] = 1 - current_status

                # Update the state based on the action
                self.update_state(action)

                # Calculate coherence after all actions for this node
                new_coherence_measure = self.calculate_coherence()

                # Save the results for this action
                results_for_node.append((action, new_coherence_measure))

                # Reset the configuration to its original state
                self.graph.nodes[node]["active"] = current_status
                self.graph.nodes[node]["value"] = current_status

                # Find the action that maximizes coherence for this node
                best_action_for_node = max(results_for_node, key=lambda x: x[1])

                # Toggle the status of the current node based on the best action
                self.graph.nodes[node]["active"] = 1 - current_status
                self.graph.nodes[node]["value"] = 1 - current_status
                self.update_state(best_action_for_node[0])  # Update the state with the best action

                # Calculate coherence after all actions for all nodes
                final_coherence_measure = self.calculate_coherence()
                reward = self.calculate_reward(new_coherence_measure)
                done = self.is_winner()

                # Determine the winning controllers based on active status
                winning_controllers = [node for node in self.graph.nodes if self.is_node_active(node)]

                #print("Winning Configuration:", dict(self.graph.nodes(data=True)))
                print("Optimal Number of Nodes:", len(winning_controllers))
                print("Final Coherence Measure:", final_coherence_measure)
                print("Winners:", self.winning_controllers)
                print("reward",reward)

        return self.get_observation(), reward, done



    def reset(self):
        # Reset the game state
        self.coherence_measure = self.calculate_coherence()

        for node in self.graph.nodes:
            self.graph.nodes[node]["value"] = 0

        # Set the active status of all nodes in the graph during reset
        for node in self.graph.nodes:
            self.graph.nodes[node]["active"] = 1

        self.remaining_active_nodes = deque(self.get_active_nodes())

        return self.get_observation()

    def calculate_coherence(self):
        active_nodes = self.get_active_nodes()
        coherence_score = 0

        for node1 in active_nodes:
            for node2 in active_nodes:
                if node1 != node2:
                    coherence_score += self.calculate_coherence_between_nodes(node1, node2)

        return coherence_score

    def calculate_coherence_between_nodes(self, node1, node2):
        METERS_TO_MILES = 1609.34  # Conversion factor from meters to miles
        SPEED_OF_LIGHT = 3 * 1e8  # Speed of light in meters per second

        if not nx.has_path(self.graph, node1, node2):
            return 0  # No direct connection, coherence is 0

        shortest_path_length = nx.shortest_path_length(self.graph, node1, node2, weight='weight')
        latency = shortest_path_length / METERS_TO_MILES / SPEED_OF_LIGHT * 1000

        # You can add more sophisticated logic here to determine coherence based on latency

        return 1 / latency  # Simple example: higher coherence for lower latency

    """def calculate_reward(self, new_coherence_measure):
        coherence_difference = self.coherence_measure - new_coherence_measure
        return coherence_difference"""
    def calculate_reward(self, new_coherence_measure):
        coherence_difference = self.coherence_measure - new_coherence_measure
        reward_scale = 1e6  # Choisissez une valeur appropriée pour l'échelle de récompense
        reward = abs(coherence_difference / reward_scale)
        return reward

    def update_state(self, action):
        # Met à jour l'état en fonction de l'action
        node = list(self.graph.nodes())[action]
        current_status = self.graph.nodes[node].get("active", 0)
        self.graph.nodes[node]["active"] = 1 - current_status
        self.graph.nodes[node]["value"] = self.current_players[0]  # Met à jour avec l'état actuel du joueur

    def is_winner(self):
        current_coherence = self.coherence_measure

        max_coherence = current_coherence
        best_configuration = None

        for node in self.graph.nodes:
            # Toggle the status of the current node
            current_status = self.graph.nodes[node].get("active", 0)
            self.graph.nodes[node]["active"] = 1 - current_status
            self.graph.nodes[node]["value"] = 1 - current_status

            new_coherence_measure = self.calculate_coherence()

            if new_coherence_measure > max_coherence:
                max_coherence = new_coherence_measure
                best_configuration = dict(self.graph.nodes(data=True))

            # Reset the configuration to its original state
            self.graph.nodes[node]["active"] = current_status
            self.graph.nodes[node]["value"] = current_status

            # Réinitialiser la cohérence à la valeur initiale
            self.coherence_measure = current_coherence

        if best_configuration:
            # Update the winning controllers based on active status
            winning_controllers = [node for node, data in best_configuration.items() if data.get("active", 0) == 1]
            self.winning_controllers = winning_controllers

            #print("Winning Configuration:", best_configuration)
            print("Optimal Number of Nodes:", len(winning_controllers))

        return best_configuration is not None

    def is_node_active(self, node):
        if node not in self.graph.nodes:
            raise ValueError(f"Node {node} not found in the graph.")

        is_active = self.graph.nodes[node].get("active", 0)
        return is_active == 1

    def get_active_nodes(self):
        return [node for node in self.graph.nodes if self.is_node_active(node)]

    def legal_actions(self):
        return list(range(len(self.graph.nodes)))

    def render(self):
        #print("Nodes:", list(self.graph.nodes()))
        #print("Coherence Measure:", self.coherence_measure)
        print("Active Nodes:", self.legal_actions())

    def get_observation(self):
        nodes_list = list(self.graph.nodes())
        node_to_index = {node: index for index, node in enumerate(nodes_list)}
        observation = np.array([[float(node_to_index[node]), float(self.coherence_measure)] for node in nodes_list])
        observation = observation[:, :, np.newaxis]

        return observation




