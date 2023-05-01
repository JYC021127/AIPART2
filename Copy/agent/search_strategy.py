# Planning to write most of the functions here, not sure if this will work 


'''
To do: 
    1. Initialize root node (state of the game)
    2. Also need to decide on the node structure (classes)
    2.1. UCB1 formula for a selection policy 
    2.2. Required to write some sort of expansion algorithm 
    3. Simulation (random?):
        3.1. Need a function representing game termination (hopefully can copy teachers code from referee)
    4. Choose the best action? 
    5. backpropogating algorithm (also need the result of the simulation) 
    6. ALgorithm that finds all the legal moves (Probably have to randomly select one of the moves from this list)
    7. Also need to limit how many simulations are done each move i
 
    
    Also: 
    - Suppose an action was done, we need to know what the board would look like after the action was applied (can possibly
    copy code from referee?)
'''



from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir

import math
from copy import deepcopy
import random 



class NODE:
    # These are independent and not shared
    def __init__(self, state, action = None, parent = None, children = None, wins = 0, playouts = 0):
        self.state = state # Current configuration of the board (dictionary format) 
        self.action = action # Action parent node took to get to current state 
        self.parent = parent # Parent node
        self.children = children if children is not None else [] # List of children nodes, not sure whether this works yet. 
        # if children is None, assign an empty list to the field
        self.wins = wins # Number of wins
        self.playouts = playouts # Number of relating sumulations / playouts 
        

    # Methods of the node: i.e. Appending children to node, 
    
    # Method that adds child node to self.children (list of children)
    def add_child(self, child):
        self.children.append(child)
    


'''
May also need a class representing the grid: with the number of red or blues? The total power of the current board? The total number 
of turns from the empty board state?
Not sure whether this is necessary
'''
class Board:




# Function that takes a board grid state (dictionary), and the colour of player as input and ouputs all possible legal actions the player can make

'''Note the legal actions are:
    - Spawning while total power < 49
    - Spreading actions that result from our current nodes  
    - Spreading as we know

We choose a random action out of the legal actions when simulating i think
'''

def get_legal_actions(grid_state, player_colour):







# Function that takes a board grid state as input, and checks whether the game has ended?

'''
Game has ended when:
    - When there exist only one colour on the board
    - When red = 0 or blue = 0:
        - draw if red and blue both equal 0
        - red wins if blue = 0
        - blue wins if red = 0
    - When there has been a total of 343 turns without the winner declared

Teachers code for this part is in referee/game/board.py "game_over" function (around line 169 - 187) 
'''
def endgame(grid_state):



