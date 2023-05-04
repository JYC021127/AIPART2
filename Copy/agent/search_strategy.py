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

    avoid obviously bad moves, i.e. ones that kill yourself. R1 spread towards R6 kills both of them, same as spawning right next to enenmy
'''


from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir

from math import *
from copy import deepcopy
import random

MAX_POWER = 49  # Max total power of a game
MAX_TURNS = 343 # Max total turns in a game
MAX_DEPTH = 10  # This shouldn't be here I think, the idea of the simulation is that you play the game until someone wins, Its not very costly, cuz the branching facor is always 1
SIMULATIONS = 500

# Class representing Nodes of the Monte Carlo Tree
class NODE:
    def __init__(self, board, action = None, parent = None, children = None, total = 0, wins = 0, playouts = 0):
        self.board = board # Current grid_state, may need to change the data structure to cater for HexPos since we are using teachers actions code 
        self.action = action # Action parent node took to get to current state 
        self.parent = parent # Parent node
        self.children = children if children is not None else [] # List of children nodes, defined as empty list if children is None, not sure whether this works yet. 
        self.total = total # Initial total moves of the game state? 
        self.wins = wins # Number of wins
        self.playouts = playouts # Number of relating sumulations / playouts 
         


    # Method that adds child node to self.children (list of children)
    '''
    O(1) generally, just adding an element into a list
    '''
    def add_child(self, child):
        self.children.append(child)
   
    # Function that calculates all the legal moves a player can do (can be found by even vs odd of self.board.turns since red starts game first)
    '''
    O(n^2) generally, worse case O(n) , 49 iterations , accessing dictionary is constant on average, appending to list is O(1) on average, 
    '''
    def get_legal_actions(self): 

        legal_actions = []
        board = self.board
        flag = 0
       
        # Using flag to flag whether board is eligible for spawning
        if board.total_power < MAX_POWER:
            flag = 1
        
        # While total power of board state < 49, all empty positions are valid spawn actions, but we can also spread
        for x in range(0, 7):
            for y in range(0, 7):
                coord = (x, y)
                if flag: # flag? 
                    # spawn action
                    if coord not in board.grid_state:
                        legal_actions.append(SpawnAction(HexPos(coord[0], coord[1])))
                    
                # spread action, this can happen independent to the total power of the board state 
                if coord in board.grid_state: # Check whether value inside, otherwise it raises key error i think
                    if board.grid_state[coord][0] == board.player_turn:
                        legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.Up))
                        legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.UpRight))
                        legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.DownRight))
                        legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.Down))
                        legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.DownLeft))
                        legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), HexDir.UpLeft))


        
        ## use this, do you mean like legal_actions[random(len(legalactions))]
        self.total = len(legal_actions)
        # need to write a function to choose which child nodes to store
            # pop and append child nodes in self.children
        legal_actions.clear()

        return legal_actions


    # Function that generates a new child node of the selected node (the selection policy was random)
    def expand(self):
        # Get a random action from a list of legal actions (when we apply heuristic, we avoid picking actions that are stupid (killing own piece / spawning next to opponent))
        random_action = random.choice(self.legal_actions)

        # Create / Deepcopy original grid and apply the random action
        next_grid = deepcopy(self.board.grid_state)
        next_grid.apply_action(random_action)

        # Initialize new child and add into into children list of self / parent
        child = NODE(board = next_grid, action = random_action, parent = self, children = None)
        self.children.append(child)
        return child


   
    # Function that checks whether a node is fully explored based on playouts and total children(True if fully explored, False if not fully explored) 
    def fully_explored(self):
        return self.playouts >= self.total


    # calculate UCB1 score
    def UCB(self):
        c = 2   # just testing out
        value = self.wins/self.playouts
        return value + c * sqrt(log(self.parent.playouts)/self.playouts)

        
    # returns the child of the node with the largest ucb score
    def largest_ucb(self):
        flag = 0 # used for the first child
        largest = 0
        largest_child = None
        for child in self.children:
            if flag == 0:
                largest = child.UCB()
                largest_child = child
                flag = 1
            # select child with larger score
            else:
                if child.UCB() == 0 or child.UCB() > largest:
                    largest = child.UCB()
                    largest_child = child
        return largest_child

            
# Class representing information relating to the grid
class BOARD:
    
    def __init__(self, state):
        self.grid_state = grid_state
        self.num_blue = num_blue
        self.num_red = num_red
        self.total_power = total_power
        self.turns = turns

    # Refer to teachers code, not that HexPos is used, not purely coordinates:colour, power
    # We need to keep track of what dictionary we are using, may need to deep copy, because dictionaries are like pointers to arrays in C
    def apply_action(self, action: Action): # turn() function used in referee > game > __init__.py
        match action: 
            case SpawnAction():
                self.resolve_spawn_action(action)
            case SpreadAction():
                self.resolve_spread_action(action)
            case _:
                raise ValueError("This isn't supposed to happen. The only 2 actions should be Spread and Spawn ") # Not sure whether Raise ValueError works


    # Function that takes an SpawnAction as input and updates the board accordingly
    def resolve_spawn_action(self, action: SpawnAction):

        # self.validate?. need to make sure argument types are correct, and board position is not currently occupied?
        
        cell = action.cell   
        
        # Can't spawn when total power of board is already at max power
        if (self.total_power >= MAX_POWER):
            raise ValueError("Not supposed to happen? L95. Max power already reached") 
        
        colour = self.player_turn
        from_cell = action.cell
        
        # Update dictionary with new spawn action
        coordinates = (from_cell.r, from_cell.q)
        self.grid_state[coordinates] = (colour, 1)

        # Update grid_state / board fields
        if colour == 'R':
            self.num_red += 1:
        else:
            self.num_blue += 1

        self.total_power += 1
        self.turns += 1


    # Function that takes a SpreadAction as input and updates the board accordingly (torus structure, and tuple addition)
    def resolve_spread_action(self, action: SpreadAction):
        # self.validate 

        # Setup: current colour turn, get hex position (internet seems to say we can use hexpos without modifying) and direction
        colour = self.player_turn
        from_cell, dir = action.cell, action.direction

        if (self.board[from_cell][0] != colour):
            raise ValueError("Spread origin node doesn't belong to the current colour")
        
        # Delete cell where spreading originates and update board fields (number of nodes of the colour is reduced by 1)
        del self.board[from_cell]
        if colour == 'R':
            self.num_red -= 1
        else:
            self.num_blue -= 1

        # Update the board_grid grid_state
        for i in range(self.board[from_cell][1]):
            # Location of coordinate spread position: make sure coordinate in torus structure 
            spread_coord = add_tuple(from_cell, mult_tuple(dir, i + 1))
            spread_coord = fix_tuple(spread_coord)

            # If coordinate to spread inside dictionary: delete node if power already 6, otherwise change colour and add one to original power 
            if spread_coord in self.board:

                # If spread_coord has a power 6 node, it will disappear. Update board fields accordingly
                if self.board[spread_coord][1] == 6:
                    if eval_colour(spread_coord) == 'R':
                        self.num_red -= 1
                    else:
                        self.num_blue -= 1

                    self.total_power -= 7 # power lost is 1 + 6
                    del self.board[spread_coord]

                # Otherwise, spread action: original colour changes and original power += 1 
                else:
                    self.board[spread_coord] = (colour, self.board[spread_coord][1] + 1)

                    # Enemy node was eaten while spreading (total power doesn't change)
                    if colour != eval_colour(spread_coord):
                        if colour == 'R':
                            num_red += 1
                            num_blue -= 1
                        else:
                            num_red -= 1
                            num_blue += 1
                        

            # Otherwise, coordinate to spread no currently occupied, so spawn a new node (total power doesn't change)
            else:
                self.board[spread_coord] = (colour, 1)
                if colour == 'R':
                    num_red += 1
                else:
                    num_blue += 1

        self.turns += 1

    

    # Assuming coordinate is inside the board, returns the colour of the coordinate on the board
    def eval_colour(self, coordinate):
        return self.board[coordinate][0]

    # Function that does vector addition for 2 coordinates
    def add_tuple(self, a, b):
        return (a[0] + b[0], a[1] + b[1])

    # Function that does scalar multiplication on the tuple a
    def mult_tuple(self, a, scalar):
        return (scalar * a[0], scalar * a[1])
    
    # Function that "fixes" the tuple so that the tuple remains in the torus structure
    def fix_tuple(self, a):
        return (a[0] % 7, a[1] % 7)

    
    @property 
    # Function that evalutes the board turns and returns the player colour that is to play in the current turn (Red: even, Blue: odd)
    '''
    O(1), just accessing stuff
    '''
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







# perform monte carlo tree search: Initialize, Select, Expand, Simulate, Backproporgate
def mcts(node, max_iterations):
    count = 0
    while count < max_iterations:


        # Traverse tree and select best node based on UCB until reach a node that isn't fully explored
        while not node.board.game_over and node.fully_explored():
            node.playouts += 1 # Don't think this is needed, this shouldn't be here
            node = node.largest_ucb()

        # is a leaf node (expansion)
        if not node.board.game_over and not node.fully_explored():
            node = expand(node) # <- find all possible moves & setting U(n) and N(n) = 0

        # Simulation (only simulate nodes, where there still exist unexplored children)
        simulate(node)
        backpropogate(value)


        if node.children is None:
            value = simulate(node)
            # backpropagation
            backpropagate(value)

        count += 1
    return best_action() # need to write function for this:  


# expansion, store partial actions as child nodes
def expand(self):
    # Get a random action from a list of legal actions (when we apply heuristic, we avoid picking actions that are stupid (killing own piece / spawning next to opponent))
    random_action = random.choice(self.legal_actions)
    next_grid = deepcopy(self.board.grid_state)
    next_grid.apply_action
    


# simulation, play randomly
def simulate(node):
    depth = 0
    while not node.board.game_over() and depth < MAX_DEPTH:
        # might wanna fix this line, get_legal_actions no longer return a list of actions
        # need random policy function for what random actions we want?
        actions = node.get_legal_actions()
        ####
        random_index = random.randint(0, len(actions)-1)
        random_action = actions[random_index]
        play(random_action) # need to write this: play the action and change the node to the result of action
        depth += 1
