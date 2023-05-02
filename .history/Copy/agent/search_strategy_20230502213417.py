# Planning to write most of the functions here, not sure if this will work 


'''
To do: 
    1. Initialize root node (state of the game)
    2. Also need to decide on the node structure (classes)
        2.1. UCB1 formula for a selection policy 
        2.2. Required to write some sort of expansion algorithm 
    3. Simulation (random?, or choose based on heuristics):
        3.1. Need a function representing game termination (hopefully can copy teachers code from referee)
        3.2. Also need delete relvant trees after simulation is completed 
    4. Choose the best action? 
    5. backpropogating algorithm (also need the result of the simulation) 
        5.1. Function that works out the winner of the game 
    6. ALgorithm that finds all the legal moves (Probably have to randomly select one of the moves from this list)
    7. Also need to limit how many simulations are done each move i
 
    
    Note that the root of the monte carlo tree is continuously changing, the root becomes some child of the monte carlo tree
    after each move of the game. The initial "parent / root" and other "irrelevant children" can be "discarded" since the game 
    has already advanced past that state (this is to save memory)

    Also: 
    - Suppose an action was done, we need to know what the board would look like after the action was applied (can possibly
    copy code from referee?)
'''


from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir

import math
from copy import deepcopy
import random 

MAX_POWER = 49
MAX_TURNS = 343

class NODE:
    # These are independent and not shared
    def __init__(self, board, action = None, parent = None, children = None, wins = 0, playouts = 0):
        self.board = board # Current configuration of the board (dictionary format) 
        self.action = action # Action parent node took to get to current state 
        self.parent = parent # Parent node
        self.children = children if children is not None else [] # List of children nodes, defined as empty list if children is None, not sure whether this works yet. 
        # if children is None, assign an empty list to the field
        self.wins = wins # Number of wins
        self.playouts = playouts # Number of relating sumulations / playouts 
        

    # Methods of the node: i.e. Appending children to node, expand node (get on of its childrens, based on a particular legal move), backpropogating, 
    
    # Method that adds child node to self.children (list of children)
    def add_child(self, child):
        self.children.append(child)
    


'''
May also need a class representing the grid: with the number of red or blues? The total power of the current board? The total number 
of turns from the empty board state? (teacher doesn't use extra memory to store num_red and num_blue,
instead it searches the whole board and looks for red and blue each time it is needed), but i guess the teacher isn't running simulations, so it doesn't matter too much for them
Not sure whether this is necessary
seems similar to teachers file "referee/game/board.py" line 48 onwards (Don't really understand the format the teacher uses in def __init__ using ":" and "->")
Python seems to allow nested classes, not 100 percent sure how to write that in at the moment 
'''
class BOARD:
    
    def __init__(self, state):
        self.grid_state = grid_state
        self.num_blue = num_blue
        self.num_red = num_red
        self.total_power = total_power
        self.turns = turns

    # Function that takes some action as input (spread or spawn), and updates the board accordingly
    # Planning to read teachers code before writing this, not sure how to include Actions inside input
    def apply_action(self, action: Action) # turn() function used in referee > game > __init__.py

        
    # Function that takes a grid_state (dictionary) as input and outputs True (game has ended) or False (game hasn't ended) 
    def game_over(self):
        return any([
            self.turns >= MAX_TURNS,
            self.num_red == 0,
            self.num_blue == 0
        ])


    # Function that takes a board grid state (dictionary), and the colour of player as input and ouputs all possible legal actions the player can make
    def get_legal_actions(self, player_colour): # Store whole action in the list i.e. SpawnAction(HexPos) and SpreadAction(Hexpos), need to look at teachers code to see how to put functions in input correctly

        legal_actions = []

        # While total power of board state < 49, all empty positions are valid spawn actions
        if (self.total_power < MAX_POWER):
            for x in range(0, 7):
                for y in range(0, 7):
                    coord = [x, y]

                    # spawn action
                    if coord not in self.grid_state:
                        legal_actions.append(SpawnAction(HexPos(coord[0], coord[1])))
                    
                    # spread action
                    else:
                        if self.grid_state[tuple(coord)][0] == player_colour:
                            legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.Up))
                            legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.UpRight))
                            legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.DownRight))
                            legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.Down))
                            legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.DownLeft))
                            legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.UpLeft))
                        
        
        # All well-defined SpreadActions are SpreadActions from each of the player_colour nodes in every direction (6 directions)
        

        return legal_actions






    # Function, unplanned yet, to work out who is the winner
    '''
def evaluate_winner(grid_state)
    if (grid_state.turns >= 343):
        return max_turn_game_result(grid_state, player_colour) # Return whether win, lose or draw

    if (grid_state.num_red == 0 or grid_state.num_blue == 0):
        if (grid_state.num_red== 0 and grid_state.num_blue ==0):
            return DRAW
        elif (grid_state.num_red == 0):
            if (player_colour == RED):
                return LOSS
            else:
                return WIN
        else:
            if (player_colour == BLUE):
                return LOSS
            else:
                return WIN
    return STILL_GOING
    '''



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



    '''Note the legal actions are:
        - Spawning while total power < 49
        - Spreading actions that result from our current nodes  
        - Spreading as we know

    We choose a random action out of the legal actions when simulating i think
    '''
