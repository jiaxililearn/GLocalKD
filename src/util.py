import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def node_iter(G):
    if nx.__version__ < '2.0.0':
        return G.nodes()
    else:
        return G.nodes

def node_dict(G):
    if nx.__version__ > '2.1.0':
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict



