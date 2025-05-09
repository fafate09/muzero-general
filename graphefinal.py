import networkx as nx
import matplotlib.pyplot as plt

def OS3EGraph(optimal_nodes=[]):
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

    pos = nx.spring_layout(g)  # you can change the layout algorithm if needed

    nx.draw(g, pos, with_labels=True, font_weight='bold', node_size=700)

    # Highlight optimal nodes
    nx.draw_networkx_nodes(g, pos, nodelist=optimal_nodes, node_color='r', node_size=700)

    plt.show()

# Example usage:
optimal_nodes = ["Seattle", "Missoula", "Sunnyvale", "LosAngeles", "Tucson", "Denver", "Dallas", "Louisville",
                 "Atlanta", "Miami", "Cleveland", "Buffalo", "Pittsburgh", "Raleigh"]

OS3EGraph(optimal_nodes)
