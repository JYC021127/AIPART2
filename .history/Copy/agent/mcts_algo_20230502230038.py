# perform monte carlo tree search
from math import *
from search_strategy import *

def mcts(node, max_simulations):
    while (max_simulations):
        # is a leaf node
        # expansion
        if (node.children == []):
            expand(node) # <- find all possible moves & setting U(n) and N(n) = 0
        
        # is root node, choose best child
        else:
            # selection
            node = largest_ucb(node) # set current as the child with largest UCB
            # simulation
            simulation(node)

        # backpropagate
        while (node != None):
            backpropagate(node)

        max_simulations -= 1
    return best_action

# calculate UCB1 score
def UCB(node):
    c = 2   # just testing out
    value = node.wins/node.playouts
    return value + c * sqrt(log(node.parent.playouts)/node.playouts)

def largest_ucb(node):
    flag = 0 # used for the first child
    largest = 0
    largest_child = None
    for child in node.children:
        if flag == 0:
            largest = UCB(child)
            largest_child = child
        else:
            if UCB(child) == 0 or UCB(child) > largest:
                largest = UCB(child)
                largest_child = child
    return largest_child
