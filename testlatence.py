import networkx as nx
import geo
import networkx as nx
import json
import string
import os
import matplotlib.pyplot as plt
def OS3EGraph():
    g = nx.Graph()
    g.add_nodes_from(["Vancouver", "Seattle"])
    g.add_nodes_from(["Missoula", "Minneapolis", "Chicago"])
    g.add_nodes_from(["SaltLakeCity"])
    g.add_nodes_from(["Portland", "Sunnyvale"])
    g.add_nodes_from(["LosAngeles", "Tucson", "ElPaso"])
    g.add_nodes_from(["Denver", "Albuquerque"])
    g.add_nodes_from(["KansasCity", "Dallas", "Houston"])
    g.add_nodes_from(["Jackson", "Memphis", "Nashville"])
    g.add_nodes_from(["BatonRouge", "Jacksonville"])
    g.add_nodes_from(["Indianapolis", "Louisville", "Atlanta"])
    g.add_nodes_from(["Miami"])
    g.add_nodes_from(["Cleveland", "Buffalo", "Boston", "NewYork", "Philadelphia", "Washington"])
    g.add_nodes_from(["Pittsburgh", "Ashburn", "Raleigh"])
    
    g.add_edges_from([("Vancouver", "Seattle"),
                      ("Seattle", "Missoula"), ("Missoula", "Minneapolis"), ("Minneapolis", "Chicago"),
                      ("Seattle", "SaltLakeCity"),
                      ("Seattle", "Portland"), ("Portland", "Sunnyvale"),
                      ("Sunnyvale", "LosAngeles"), ("LosAngeles", "Tucson"), ("Tucson", "ElPaso"),
                      ("SaltLakeCity", "Denver"), ("Denver", "Albuquerque"), ("Albuquerque", "ElPaso"),
                      ("Denver", "KansasCity"), ("KansasCity", "Dallas"), ("Dallas", "Houston"),
                      ("Houston", "Jackson"), ("Jackson", "Memphis"), ("Memphis", "Nashville"),
                      ("Houston", "BatonRouge"), ("BatonRouge", "Jacksonville"),
                      ("Chicago", "Indianapolis"), ("Indianapolis", "Louisville"), ("Louisville", "Nashville"),
                      ("Nashville", "Atlanta"),
                      ("Atlanta", "Jacksonville"),
                      ("Jacksonville", "Miami"),
                      ("Chicago", "Cleveland"), ("Cleveland", "Buffalo"), ("Buffalo", "Boston"),
                      ("Boston", "NewYork"), ("NewYork", "Philadelphia"), ("Philadelphia", "Washington"),
                      ("Cleveland", "Pittsburgh"), ("Pittsburgh", "Ashburn"), ("Ashburn", "Washington"),
                      ("Washington", "Raleigh"), ("Raleigh", "Atlanta")])

    return g
def write_json_file(filename, data):
    '''Given JSON data, write to file.'''
    json_file = open(filename, 'w')
    json.dump(data, json_file, indent = 4)


def read_json_file(filename):
    input_file = open(filename, 'r')
    return json.load(input_file)


METERS_TO_MILES = 0.000621371192
LATLONG_FILE = "latlong.json"


def lat_long_pair(node):
    return (float(node["Latitude"]), float(node["Longitude"]))


def dist_in_miles(data, src, dst):
    '''Given a dict of names and location data, compute mileage between.'''
    src_pair = lat_long_pair(data[src])
    src_loc = geo.xyz(src_pair[0], src_pair[1])
    dst_pair = lat_long_pair(data[dst])
    dst_loc = geo.xyz(dst_pair[0], dst_pair[1])
    return geo.distance(src_loc, dst_loc) * METERS_TO_MILES


"""def OS3EWeightedGraph():
    data = {}
    g = OS3EGraph()
    longit = {}
    lat = {}
    # Get locations
    if os.path.isfile(LATLONG_FILE):
        print("Using existing lat/long file")
        data = read_json_file(LATLONG_FILE)
    else:
        return g

    for node in g.nodes():
        latit = float(data[node]["Latitude"])
        lon = float(data[node]["Longitude"])
        lat[node] = latit
        longit[node] = lon
    nx.set_node_attributes(g, lat, 'Latitude')
    nx.set_node_attributes(g, longit, 'Longitude')

    # Append weights
    for src, dst in g.edges():
        g[src][dst]["weight"] = dist_in_miles(data, src, dst)
        #print "%s to %s: %s" % (src, dst, g[src][dst]["weight"])
        print(f"Edge: {src} -> {dst}, Weight: {g[src][dst]['weight']}")
    return g
"""
def OS3EWeightedGraph():
    data = {}
    g = OS3EGraph()
    longit = {}
    lat = {}
    # Get locations
    if os.path.isfile(LATLONG_FILE):
        print("Using existing lat/long file")
        data = read_json_file(LATLONG_FILE)
    else:
        return g

    for node in g.nodes():
        latit = float(data[node]["Latitude"])
        lon = float(data[node]["Longitude"])
        lat[node] = latit
        longit[node] = lon
    nx.set_node_attributes(g, lat, 'Latitude')
    nx.set_node_attributes(g, longit, 'Longitude')

    # Append weights (distances)
    for src, dst in g.edges():
        # Calculate the distance between nodes
        distance = dist_in_miles(data, src, dst)
        g[src][dst]["weight"] = distance
        # Print the latencies (distances) to see if they are correct
        print(f"Latency between {src} and {dst}: {distance:.2f} miles")

    return g

# Appeler la fonction pour afficher les latences
weighted_graph = OS3EWeightedGraph()

# Vous pouvez également visualiser le graphe pondéré après avoir calculé les latences
def visualize_weighted_graph(graph):
    pos = nx.spring_layout(graph)  # Layout pour une meilleure visualisation
    weights = nx.get_edge_attributes(graph, 'weight')

    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)
    print("Edge Weights:")
    for edge, weight in weights.items():
        print(f"{edge}: {weight}")

    plt.show()

# Visualiser le graphe pondéré avec les latences calculées
visualize_weighted_graph(weighted_graph)

def visualize_weighted_graph(graph):
    pos = nx.spring_layout(graph)  # Layout pour une meilleure visualisation
    weights = nx.get_edge_attributes(graph, 'weight')

    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)
    print("Edge Weights:")
    for edge, weight in weights.items():
        print(f"{edge}: {weight}")

    plt.show()
OS3EWeightedGraph()
# Utiliser la fonction OS3EWeightedGraph() pour obtenir le graphe pondéré
weighted_graph = OS3EWeightedGraph()

# Visualiser le graphe pondéré
visualize_weighted_graph(weighted_graph)
