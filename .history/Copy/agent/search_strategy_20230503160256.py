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

from math import *
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
    

# Class representing information relating to the grid
class BOARD:
    
    def __init__(self, state):
        self.grid_state = grid_state
        self.num_blue = num_blue
        self.num_red = num_red
        self.total_power = total_power
        self.turns = turns

    # Function that takes some action as input (spread or spawn), and updates the board accordingly
    # Planning to read teachers code before writing this, not sure how to include Actions inside input
    # Wonder whether this can be imported from the teachers code, but it seems quite hard. Don't really know what
    # the teacher is writing. Perhaps just copy the idea of the teachers code
    def apply_action(self, action: Action): # turn() function used in referee > game > __init__.py
        match action: 
            case SpawnAction():
                self.resolve_spawn_action(action)
            case SpreadAction():
                self.resolve_spread_action(action)
            case _:
                raise ValueError("This isn't supposed to happen. The only 2 actions should be Spread and Spawn ") # Not sure whether Raise ValueError works




    # Need to somehow know that player colour, the action doesn't provide any information about the colour to apply
    # But we can use that red is always turn 0, blue is always turn 1 -> red is always even, and blue is always odd turn -> turns % 2 == 0 or not to determine colour

    # Function that takes an SpawnAction as input and updates the board accordingly
    def resolve_spawn_action(self, action: SpawnAction):
        # self.validate?
        
        cell = action.cell   
        
        if (self.total_power >= MAX_POWER):
            raise ValueError("Not supposed to happen? L95. Max power already reached") 
        
        colour = self.player_turn()
        
        # Now we update the grid accordinly



    # Function that takes a SpreadAction as input and updates the board accordingly
    def resolve_spread_action(self, action: SpreadAction):
        # self.validate 

        colour = self.player_turn()
        from_cell, dir = action.cell, action.direction
         

# Function that evalutes the board turns and returns the player colour that is to play in the current turn (Red: even, Blue: odd )
    def player_turn(self) -> str:
        # Red's turn when total turns is even
        if self.turns % 2 == 0:
            return 'r'
        else: #'B' plays on odd turn
            return 'b'
   


    # Function that takes a grid_state (dictionary) as input and outputs True (game has ended) or False (game hasn't ended) 
    '''
    O(1), just accessing stuff
    '''
    def game_over(self):
        return any([
            self.turns >= MAX_TURNS,
            self.num_red == 0,
            self.num_blue == 0
        ])

    # Function that takes in a grid_state (dictionary) as input and outputs the colour of the winner ("R" or "B") 
    # Make sure that this function is only run after game_end condition is satisfied. There are only 4 conditions of end_game that determine winner
    '''
    O(1), just accessing stuff and comparing them
    '''
    def winner(self):
        # If board reached max number of turns (343 turns), the winner is the colour with the most power 
        if self.turns >= MAX_TURNS:
            return max_power_colour()
        # Otherwise, empty board represents draw and the colour without nodes on the board is the LOSING colour 
        else:
            if (self.num_red == 0 and self.num_blue == 0):
                return None # need a colour here that represents draw, perhaps use "W" as a global constant ("WHITE"), neither blue or red has won
            elif self.num_red == 0:
                return 'B'
            else: # self.num_blue == 0
                if self.num_blue != 0:
                    raise ValueError("Something is wrong, perhaps the ending condition is not satisfied")
                return 'R'


    # Function that takes a board grid state (dictionary), and the colour of player as input and ouputs all possible legal actions the player can make
    def get_legal_actions(self): # Store whole action in the list i.e. SpawnAction(HexPos) and SpreadAction(Hexpos), need to look at teachers code to see how to put functions in input correctly

        legal_actions = []

        # While total power of board state < 49, all empty positions are valid spawn actions
        for x in range(0, 7):
            for y in range(0, 7):
                coord = (x, y)
                if (self.total_power < MAX_POWER): # flag? 
                    # spawn action
                    if coord not in self.grid_state:
                        legal_actions.append(SpawnAction(HexPos(coord[0], coord[1])))
                    
                # spread action
                else:
                    if self.grid_state[coord][0] == self.player_turn:
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
# perform monte carlo tree search
def mcts(node, max_iterations):
    while (max_iterations):
        # set up tree
        while not node.board.game_over():
            # is a leaf node (expansion)
            if node.children == []:
                expand(node) # <- find all possible moves & setting U(n) and N(n) = 0
        
            # is root node, choose best child (selection)
            else:
                node = largest_ucb(node) # set current as the child with largest UCB

        # simulation 
        while not node.board.game_over():
            simulation(node)

        # backpropagate
        while node:
            backpropagate(node)

        max_simulations -= 1
    return best_action # need to write function for this 

# calculate UCB1 score
def UCB(node):
    c = 2   # just testing out
    value = node.wins/node.playouts
    return value + c * sqrt(log(node.parent.playouts)/node.playouts)

# returns the child of the node with the largest ucb score
def largest_ucb(node):
    flag = 0 # used for the first child
    largest = 0
    largest_child = None
    for child in node.children:
        if flag == 0:
            largest = UCB(child)
            largest_child = child
            flag = 1
        else:
            if UCB(child) == 0 or UCB(child) > largest:
                largest = UCB(child)
                largest_child = child
    return largest_child

# expansion, store all actions as child nodes
def expand(node):
    actions = node.board.get_legal_actions(node.board)
    for action in actions:
        node.children.append(action)
        node.wins = 0
        node.playouts = 0

