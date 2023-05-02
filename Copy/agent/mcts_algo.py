# perform monte carlo tree search
def mcts(node):
    while (node != terminal node) { # maybe use evaluate_winner(node.grid_state) here?
        # is a leaf node
        if (node.children == []) {
            if (node.playouts == 0) {
                # simulation(node)
            }
            else {
                # expand(node) <- find all possible moves & setting U(n) and N(n) = 0
                node = node.child # selecting the first child node i think??
                simulation(node)
                backpropagate(node)
            }
        }
        else {
            node = largest_ucb(node) # set current as the child with largest UCB
        }
    }

# calculate UCB1 score
def UCB(node):
    c = 2   # just testing out
    value = node.wins/node.playouts
    return value + c * sqrt(log(node.parent.playouts)/node.playouts)

