
# Need to create a ranking system for obvious actions, return the action that has the largest spread action (highest power spread action)
# Obvious moves: killing high power enenmy, spreading own high power cell if it is in danger
# Right now its not spawning next to each other kinda

# Other problems: spreading towards nowhere?, which does nothing

from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir


from .utils import render_board

from math import *
from copy import deepcopy
import random
import time

MAX_ITERATIONS = 100
MAX_POWER = 49  # Max total power of a game
MAX_TURNS = 343 # Max total turns in a game, This might be 342, since the teacher's turn starts at 1, and we start at 0, but it shouldn't matter too much (Actually, i need to think about this a bit more)

class DIR:
    coord = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]
    hex_dir = [HexDir.Up, HexDir.UpRight, HexDir.DownRight, HexDir.Down, HexDir.DownLeft, HexDir.UpLeft]
    
# Class representing Nodes of the Monte Carlo Tree
class NODE:
    def __init__(self, board, action = None, parent = None, children = None, total = 0, wins = 0, playouts = 0):
        self.board = board # Infexion grid and relevant information relating to the grid
        self.action = action # Action parent node took to get to current state 
        self.parent = parent # Parent node
        self.children = children if children is not None else [] # Children is children if not, otherwise it is empty list 
        self.total = total # total number of possible moves 
        self.wins = wins # Number of wins
        self.playouts = playouts # Number of relating sumulations / playouts 

 
    # Method that adds child node to self.children (list of children)
    '''
    O(1) generally, just adding an element into a list
    '''
    def add_child(self, child):
        self.children.append(child)   
        

    '''
    Function that generates a new child node of the selected node (the selection policy is based on specialized game knowledge):
        - Avoid picking unlikely actions, spawning on random region of the board, Spawning next to opponent, Killing own piece (killing own power 6)
        - Favour exploring regions that are promising for both colours: Spawning in clusters, spreading if opponent is reachable
    O(n^2) + O(n^2) = O(n^2), getting all the legal actions is dictionary size + deepcopy is also size of dictionary
    '''
#    ##### might need to check the complexity again
#    #### Oh lol, when did u change this. I thought we were only expanding one child each time haha, there's no point using total then here
#    ### If u use this, limiting the branching factor won't work. Did u want to expand multiple nodes at once?
#    def expand(self):
#        total = self.total 
#        count = 0
#
#        # Randomly selecting an action from a list of favourable / likely actions
#        actions = self.board.get_legal_actions 
#
#        # will keep finding the best action if child already exists in children
#        while count < total:
#            random_action = self.board.heuristic(actions)
#
#            if self.child_exists(random_action):
#                print("This shouldn't be run I think")
#                actions.remove(random_action)
#
#            else:
#                # Create / Deepcopy original grid and apply the random action to the board
#                board = deepcopy(self.board)
#                board.apply_action(random_action)
#
#                # Initialize new child and add into into children list of self / parent.
#                child = NODE(board = board, action = random_action, parent = self, children = None, total = len(board.get_legal_actions))
#
#                # Add child to self.children list
#                self.add_child(child)
#                # Return the child node (we need to simulate this node)
#                return child
#            
#            count += 1
#        # del actions # Apparently this is not needed, but can be used 


    def expand(self):
        
        # Randomly selecting an action from a list of favourable / likely actions
        actions = self.board.get_legal_actions 
        
        # Function that deletes all actions that are already child nodes
        actions = self.clean_actions(actions) 

        random_action = self.board.heuristic(actions)
        # del actions # Apparently this is not needed, but can be used 
       

        # Create / Deepcopy original grid and apply the random action to the board
        board = deepcopy(self.board)
        board.apply_action(random_action)

        # Initialize new child and add into into children list of self / parent. (Im not sure what you mean by total, but im assuming the total number of possible children nodes)
        child = NODE(board = board, action = random_action, parent = self, children = None, total = len(board.get_legal_actions))

        # Add child to self.children list
        self.add_child(child)
        
        # Return the child node (we need to simulate this node)
        return child


    # Function that "cleans" the actions list so that all actions are non-expanded nodes (avoids running heuristic too many times)
    # AI assisted function (more concise): actions list where action is not any of the child.action in self.children
    def clean_actions(self, actions):
        return [action for action in actions if action not in [child.action for child in self.children]] 


    # Function that simulates / rollout of the node and returns the colour of the winner
    '''
    O(n^2) + (depth of game) * O(n^2), deep copy + expand * depth of tree. There is no branching factor because it is a "stick" that is repeately deleted
    Time complexity is mostly influenced by the length of the game
    '''
    def simulate(self):

        # Create a deep copy of the node we can modify (independent of the node on the tree)
        node = deepcopy(self)

        # Avoiding errors
        node.parent = None
        node.children = None
        node.action = None
        node.total = 0
  
        # While not game over, keep playing a move
        while not node.board.game_over and not node.board.too_much_adv:
            actions = node.board.get_legal_actions

            # Simulate with choices based on a light weight heuristic  
            random_action = node.board.light_heuristic(actions)
            
            #del actions # Apparently not needed
            node.board.apply_action(random_action)

        # Evaluate winner, after game ending condition satisfied (escaped while loop)
        winner = node.board.winner 
        #del node # May not be needed, python garbage collector can deal with it

        # Winning colour
        return winner


    # Function that backpropogates the result (either 'r' or 'b' has won), updating the wins and the playouts
    '''
    O(d), where d is the depth of the tree. Updating playouts are generally O(1) 
    '''
    def backpropogate(self, result):
        # Set node to itself
        node = self

        # While node is not None, playouts += 1 , if player turns == winning colour, wins += 1 
        while node is not None:
            node.playouts += 1
            # If result / winner colour == player_turn colour, += 1 
            if result == node.board.player_turn:
                node.wins += 1
            node = node.parent  

     
    # Function that checks whether a node is fully explored based on playouts and total children(True if fully explored, False if not fully explored) 
    '''
    O(1)
    '''
    @property
    def fully_explored(self):
        return self.playouts >= self.total


    # OPTIMIZE
    # Function that checks whether a node is explored "enough" (This can be tweaked to see what works the best)
    '''
    O(1), accessing and calculation
    '''
    @property 
    def explored_enough(self):       
        # Explored enough, if 1/4 of of total branches are searched and more than 10 branches already searched
        if len(self.children) >= (self.total / 5) and len(self.children) >= 20:
            return True

        # Explored enough, if every branch is explored at least once
        if self.fully_explored:
            return True
        return False


    # calculate UCB1 score
    '''
    O(1)
    '''
    def UCB(self, c = 2):
        # If node has no playouts, it is automatically infinity due to right sum "exploding"
        if self.playouts == 0:
            return float('inf')
        else:
            value = self.wins/self.playouts
            return value + c * sqrt(log(self.parent.playouts)/self.playouts)# + self.board.board_score()/(self.playouts * 5)
    


    # Function that takes a node as input and returns the child node with the largest UCB score
    '''
    O(n), where n is the number of children. Calculations are generally O(1) 
    '''
    def largest_ucb(self):
        # Initialize / Setup
        largest = float('-inf')
        largest_child = None

        for child in self.children:
            if child.UCB() > largest:
                largest = child.UCB()
                largest_child = child
        return largest_child


    # Function that loops over the children of the node (generally the root node), and outputs the best action: based on total playouts at the moment
    '''
    O(n), where n is the number of children, accessing playouts is generally O(1)
    '''
    def best_final_action(self):
        # self.children is list of child nodes, lamda takes child nodes and returns playouts, max returns the child of the most playouts
        best_child = max(self.children, key = lambda child: child.playouts)

        # Action applied to parent grid_state to get to child grid_state
        return best_child.action


    # function that checks if an action already exists in children list
    def child_exists(self, action):
        for child in self.children:
            if action == child.action:
                return True
        return False
    

    # For debugging purposes: function that prints the fields of the NODE class
    def print_node_data(self, depth = 0):
        print("Printing Node data:")
        print(f"Node depth is {depth}")
        print(f"The node itself is {self} \nIt looks like this:")
        print(render_board(self.board.grid_state, ansi = True))
        print(f"The action is {self.action}")
        print(f"The parent node is {self.parent}")
        print(f"There are {len(self.children)} children, and the children nodes are {self.children}")
        print(f"The total legal moves of the node are {self.total}")
        print(f"The number of wins of the node is {self.wins}")
        print(f"The number of playouts of the node is {self.playouts}")


    # For debugging purposes: function that prints the fields of every NODE on the tree (Also counts the number of nodes, and prints the node depth) 
    # AI assisted function
    def print_whole_tree_node_data(self, depth = 0, counter = None):
        if counter is None:
            counter = {'count': 0}
        
        self.print_node_data(depth)
        counter['count'] += 1
        if self.children:
            for child in self.children:
                child.print_whole_tree_node_data(depth + 1, counter)

        if depth == 0:
            print(f"Total number of nodes in the tree: {counter['count']}")
        
        return counter['count']

    # For debugging purposes: function that prints the fields of every child NODE
    @property
    def print_child_node_data(self):
        count = 1
        for child in self.children:
            print(f"child {count} node data:")
            print(render_board(child.board.grid_state, ansi = True))
            child.print_node_data()
            count += 1


# Class representing information relating to the grid
class BOARD:
    def __init__(self, grid_state, num_blue = 0, num_red = 0, total_power = 0, turns = 0):
        self.grid_state = grid_state # Dictionary representing the board, coordinates : (colour, power)
        self.num_blue = num_blue # Number of blues on the board 
        self.num_red = num_red # Number of red on the board
        self.total_power = total_power # Total power of the board
        self.turns = turns # Total turns from empty board to current board 
    

    # Compares equality of instances
    def __eq__(self, other):
        if not isinstance(other, BOARD):
            raise ValueError("Comparing different classes?")
        
        return (self.grid_state == other.grid_state) and (self.num_blue == other.num_blue) and (self.num_red == other.num_red) and (self.total_power == other.total_power) and (self.turns == other.turns)


    # Function that calculates all the legal moves a player can do (can be found by even vs odd of self.board.turns since red starts game first)
    '''
    O(n^2) generally, where n = 7 representing 7 by 7 grid, accessing dictionary is constant on average, appending to list is O(1) on average, with worst case O(n) 
    '''
    @property
    def get_legal_actions(self): 

        legal_actions = []
        flag = 0
       
        # Using flag to flag whether board is eligible for spawning
        if self.total_power < MAX_POWER:
            flag = 1
        
        # While total power of board state < 49, all empty positions are valid spawn actions, but we can also spread
        for x in range(0, 7):
            for y in range(0, 7):
                coord = (x, y)

                # Spawn action allowed if total power < 49, and coordinate is not currently occupied
                if flag:  
                    if coord not in self.grid_state:
                        legal_actions.append(SpawnAction(HexPos(coord[0], coord[1])))
                    
                # Spread action allowed if origin of spread is the same as the colour of current player turn
                if coord in self.grid_state: 
                    if self.grid_state[coord][0] == self.player_turn:
                        for direction in DIR.hex_dir:
                            legal_actions.append(SpreadAction(HexPos(coord[0], coord[1]), direction))

        # return list of legal actions
        return legal_actions

    # Function that takes in a dictionary of changed nodes, "undos" the action on a dictionary
    # format of changes_dict: {"action": "spread" / "spawn" , "node_origin": (from_cell coordinate, (colour, power)), "changes":{coordinates: (prev_colour, prev_power)}}
    # Changes dictionary include the coordinate and their information before action was applied
    # Note that node_origin cell is not stored inside the changes dictionary
    def undo_action(self, changes_dict: dict):

        # If the action made was spawn, delete the spawned cell and update the board accordingly
        if changes_dict["action"] == "spawn":
            spawned_cell = changes_dict["node_origin"][0] # coordinate of node_origin
            # Get colour of to be deleted cell
            colour = self.eval_colour(spawned_cell)
            del self.grid_state[spawned_cell]
            
            # Update board information accordingly
            if colour == 'r':
                self.num_red -= 1
            else:
                self.num_blue -= 1
            self.total_power -= 1
            
        elif changes_dict["action"] == "spread":
            
            # Dictionary of changes
            changes = changes_dict["changes"]

            # Store coordinate, (colour, power)
            #before_spread_coord, before_spread_values = changes_dict["node_origin"]
            before_spread_coord = changes_dict["node_origin"][0]
            before_spread_values = changes_dict["node_origin"][1]
            
            # Get reverted colour and add cell to dictionary (revert spread_origin)
            # colour = self.eval_colour(before_spread_coord)
            # before_spread_coord, before_spread_values = changes_dict["node_origin"]
            colour = before_spread_values[0]
            self.grid_state[before_spread_coord] = before_spread_values
          
            # Update dictionary for spread origin node
            if colour == 'r':
                self.num_red += 1
            else:
                self.num_blue += 1
            self.total_power += before_spread_values[1]
          
            # Update changed cells caused by spread: Delete old values, and update new values each time
            for coordinate, value in changes.items():
                # If cell is reverted to empty, delete current cell (should be power 1) and update board accordingly
                if value is None:
                    tmp_colour = self.eval_colour(coordinate)
                    assert self.grid_state[coordinate][1] == 1, "Power of cell after spread is not 1?"
                    del self.grid_state[coordinate]
                    if tmp_colour == 'r':
                        self.num_red -= 1
                    else:
                        self.num_blue -= 1
                    self.total_power -= 1
                    continue

                # colour and power of reverted coordinate
                # colour_revert, power_revert = value
                colour_revert = value[0]
                power_revert = value[1]
                
                # Cells with power 7 disappear ("edge case"): Add this cell in first to avoid error  
                if coordinate not in self.grid_state:
                    if power_revert != 6: # Sanity check, delete afterwards: if a now empty cell isn't reverted to power 6 -> error
                        raise ValueError("coordinate that wasn't in dictionary to be reverted doesn't have power 6")
                  
                    # Update grid_state dictionary
                    self.grid_state[coordinate] = value
                    # Update board data
                    if colour_revert == 'r':
                        self.num_red += 1
                    else:
                        self.num_blue += 1
                    self.total_power += 6
                    continue # Skip current for loop iteration

                # colour and power of coordinate now (to be reverted), we know they are inside dictionary
                colour_now = self.eval_colour(coordinate)
                power_now = self.eval_power(coordinate)
                
                # Update board information, assuming coordinate was deleted
                if colour_now == 'r':
                    self.num_red -= 1
                else:
                    self.num_blue -= 1
                self.total_power -= power_now
                
                # Replace coordinate info now, with reverted coordinate info
                self.grid_state[coordinate] = value
              
                # Update board information, since coordinate value reverted was updated
                if colour_revert == 'r':
                    self.num_red += 1
                else:
                    self.num_blue += 1
                self.total_power += power_revert
        self.turns -= 1


    # Function that takes an action (either spread or spawn), and applies the action to the board / grid_state, updating the board accordingly
    # It returns a dictionary of changes applied to the cell if action_param specified
    def apply_action(self, action: Action, action_param = None): 
        # Initialize empty dictionary
        changes_dict = {}
        match action: 
            case SpawnAction():
                if action_param is not None:
                    changes_dict = self.resolve_spawn_action(action, param = action_param)
                else:
                    self.resolve_spawn_action(action)
            case SpreadAction():
                if action_param is not None:
                    changes_dict = self.resolve_spread_action(action, param = action_param)
                else:
                    self.resolve_spread_action(action)
            case _:
                raise ValueError("This isn't supposed to happen. The only 2 actions should be Spread and Spawn ")
        self.turns += 1

        if action_param is not None:
            return changes_dict

    # Function that takes an SpawnAction as input and updates the board accordingly
    '''
    O(1)
    '''
    def resolve_spawn_action(self, action: SpawnAction, param = None):

        # Could add a self.validate to confirm everything is going as expected
          
        # Can't spawn when total power of board is already at max power
        if (self.total_power >= MAX_POWER):
            raise ValueError("Not supposed to happen? L95. Max power already reached") 
        
        # Setup: determine colour of current turn, and where cell will spawn
        colour = self.player_turn
        from_cell = action.cell
        
        # Update dictionary with new spawn action
        coordinates = (int(from_cell.r), int(from_cell.q))
        self.grid_state[coordinates] = (colour, 1)

        # Update grid_state / board fields
        if colour == 'r':
            self.num_red += 1
        else:
            self.num_blue += 1
        self.total_power += 1

        if param is not None:
            dict_changes = {}
            dict_changes["action"] = "spawn"
            dict_changes["node_origin"] = (coordinates, None)
            dict_changes["changes"] = {}
            dict_changes["changes"][coordinates] = None
            return dict_changes


    # Function that takes a SpreadAction as input and updates the board accordingly (torus structure, and tuple addition)
    '''
    O(power), number of coordinates = power, where dictionary look up occurs for each of them, and fields are updated
    '''
    def resolve_spread_action(self, action: SpreadAction, param = None):
        # self.validate 
        if param is not None:
            dict_changes = {}
            dict_changes["changes"] = {}

        # Setup: current colour turn, get hex position (internet seems to say we can use hexpos without modifying) and direction
        colour = self.player_turn
        cell, dir = action.cell, action.direction
        from_cell = (int(cell.r), int(cell.q))
        dir = (int(dir.r), int(dir.q))

        # Spread origin belings to turn colour
        if from_cell in self.grid_state:
            if (self.grid_state[from_cell])[0] != colour:
                raise ValueError("Spread origin node doesn't belong to the current colour")
        
        # Store action type, and from_cell (coordinate, (colour, power))
        if param is not None:
            dict_changes["action"] = "spread"
            dict_changes["node_origin"] = (from_cell, self.grid_state[from_cell])
        

        # Update the board_grid grid_state
        for i in range((self.grid_state[from_cell])[1]):

            # Location of coordinate spread position: make sure coordinate in torus structure 
            spread_coord = self.add_tuple(from_cell, self.mult_tuple(dir, i + 1))
            spread_coord = self.fix_tuple(spread_coord)
            # If coordinate to spread inside dictionary: delete node if power already 6, otherwise change colour and add one to original power 
            if spread_coord in self.grid_state:
                # If spread_coord has a power 6 node, it will disappear. Update board fields accordingly
                if (self.grid_state[spread_coord])[1] == 6:
                    if self.eval_colour(spread_coord) == 'r':
                        self.num_red -= 1
                    else:
                        self.num_blue -= 1

                    self.total_power -= 7 # power lost is 1 + 6
                    # Add to be changed coordinate inside dict_changes if param not None, and delete the coordinate
                    if param is not None:
                        dict_changes["changes"][spread_coord] = self.grid_state[spread_coord]

                    del self.grid_state[spread_coord]

                # Otherwise, spread action: original colour changes and original power += 1 
                else:
                    # Enemy node was eaten while spreading (total power doesn't change)
                    if colour != self.eval_colour(spread_coord):
                        if colour == 'r':
                            self.num_red += 1
                            self.num_blue -= 1
                        else:
                            self.num_red -= 1
                            self.num_blue += 1

                    # Update dictionary if parameter specified
                    if param is not None:
                        dict_changes["changes"][spread_coord] = self.grid_state[spread_coord]

                    # Update num_blue and red before updating dictionary
                    self.grid_state[spread_coord] = (colour, self.grid_state[spread_coord][1] + 1)

            # Otherwise, coordinate to spread no currently occupied, so spawn a new node (total power doesn't change)
            else:
                # Update changes dict if parameter specified
                if param is not None:
                    dict_changes["changes"][spread_coord] = None

                self.grid_state[spread_coord] = (colour, 1)

                if colour == 'r':
                    self.num_red += 1
                else:
                    self.num_blue += 1
    
        # Delete cell where spreading originates and update board fields (number of nodes of the colour is reduced by 1)
        # Note this is already stored inside dict_changes["node_origin"], so it is not stored inside dict_changes["changes"]
        if colour == 'r':
            self.num_red -= 1
        else:
            self.num_blue -= 1
        del self.grid_state[from_cell]

        # Return dict changes if parameter isn't None
        if param is not None:
            return dict_changes
        
    
    # calculate the score of a board state depending on number of red and blue nodes
    def board_score(self):
        colour = self.player_turn
        num_blue = self.num_blue
        num_red = self.num_red
        score = 0

        if colour == 'r':
            score = num_red - num_blue
        else:
            score = num_blue - num_red
        
        return score


    # function that returns if there's any enemy nodes around given coord
    def check_enemy(self, coord: tuple, colour):
        # check each direction
        for dir in DIR.coord:
            tmp_coord = (coord[0] + dir[0], coord[1] + dir[1])
            tmp_coord = self.fix_tuple(tmp_coord)
            if tmp_coord in self.grid_state:
                # if neighbour is an enemy
                if (self.grid_state[tmp_coord])[0] != colour:
                    return True
        return False


    '''
    Current Heuristic:
    Obvious
    - spread action, kill enemy cells and has no enemy neighbour after spreading, high power spread action 

    Good
    - spawn action next to own with no enemy neighbour
    - spread action, kill enemy cells but has neighbour enemy cells after spreading

    Average
    - spawn action in cell with empty surrounding
    - everything else (e.g. number of enemy killed and own killed are the same)

    Bad
    - spawn action next to enemy
    - spread action and doesn't kill any enemy cell and doesn't combine power
    - number of own killed > enemy killed   
    - spreading and eating enemy cell having enemy neighbours if there's only one own cell left (basically killing yourself and ending the game)
    '''
    # Heuristic function that returns a random action amongst all legal actions from the best action list
    def heuristic(self, actions):
        good = []
        average = []
        bad = []
        colour = self.player_turn
        init_red = self.num_red
        init_blue = self.num_blue
        highestpower_cell = self.largest_power_cell(colour) # cell with highest power for that player
        copy = deepcopy(self)
        # print("\n***********start of heuristic")
        # print(f"board is {render_board(self.grid_state, ansi = True)}")
        for action in actions:
            #print(f"action is {action}")
            score = 0 # used to rank actions
            cell = action.cell
            from_cell = (int(cell.r), int(cell.q))
            power = self.eval_power(from_cell) # power of from_cell
            
            # apply action to the temporary board
            changes_dict = copy.apply_action(action, action_param = "get_actions_dict")

            # Initialize initial board, and "applied action" board
            new_red = copy.num_red
            new_blue = copy.num_blue
            flag = 0

            # Score is calculated based on change in blue and red after applying the action
            if colour == 'r':
                num_own_colour = new_red
                init_num_colour = init_red
                score = (new_red - init_red) + (init_blue - new_blue)
            else:
                num_own_colour = new_blue
                init_num_colour = init_blue
                score = (new_blue - init_blue) + (init_red - new_red)
            
            # Positive Score: Spawning, Spread that kill enemies
            if score > 0:
                if isinstance(action, SpawnAction):
                    # checking each of the neighbour cells
                    # make sure we're not spawning right next to an enemy cell
                    for dir in DIR.coord:
                        tmp_coord = (from_cell[0] + dir[0], from_cell[1] + dir[1])
                        tmp_coord = self.fix_tuple(tmp_coord)
                        if tmp_coord in copy.grid_state:
                            # neighbour is an enemy
                            if copy.eval_colour(tmp_coord) != colour:
                                #print(f"{tmp_coord} is neighbour enemy cell, can't spawn")
                                bad.append(action)
                                flag = 1
                                break

                            # neighbour is own, keep checking 
                            else:
                                flag = 2

                    # spawning in an empty surrounding
                    if flag == 0:
                        #print(f"{from_cell} has empty surrounding, average")
                        average.append(action)

                    # spawning next to own cell, with no enemy surrounding
                    elif flag == 2:
                        #print(f"{from_cell} has own surrounding, good")
                        good.append(action)

                # Spread Action
                elif isinstance(action, SpreadAction):
                    dir = action.direction
                    dir = (int(dir.r), int(dir.q))
                    flag = 1

                    # Spreading without killing ememy nodes (score > 0 since number of own colour nodes increased)
                    if flag:
                        if colour == 'r':
                            if new_blue - init_blue == 0:
                                bad.append(action)
                                flag = 0
                        else:
                            if new_red - init_red == 0:
                                bad.append(action)
                                flag = 0

                    # Spread action that kills enemy nodes (with and without consequences) 
                    if flag:
                        # from_cell's power is the largest power cell and spreading will kill some enemies
                        if from_cell in highestpower_cell:
                            return action

                        check = 0
                        enemies = 0
                        # check each cell that it will spread to
                        for i in range(power):
                            # Location of coordinate spread position: make sure coordinate in torus structure
                            spread_coord = self.add_tuple(from_cell, self.mult_tuple(dir, i + 1))
                            spread_coord = self.fix_tuple(spread_coord)

                            # if it spreads near enemy cells
                            if copy.check_enemy(spread_coord, colour) and not check:
                                # changed this part to spreading the highest power cell
                                # tmp_power = self.eval_power(spread_coord)
                                # # Action is obvious if action of spread is a high power coordinate, shortcircuit
                                # if tmp_power >= 3:
                                #     return action
                                
                                # there's only one own cell with power 1 left on the board and it spreads next to enemy cells
                                if power == 1 and init_num_colour == 1:
                                    bad.append(action)
                                    check = 1
                                    break
                                
                                else:
                                    # If can kill someone with a larger power, return the action to explore
                                    if copy.eval_power(spread_coord) > power:
                                        return action
                                    
                                    good.append(action)
                                    check = 1
                                    break

                        # Spreading that results in no neighbour enemies
                        if not check:
                            return action

            # score won't be <= 0 if it's a spawn action. The case includes: actions that don't change number of red and blue
            # This action involves spreading, whichh kills enemies and own cells (but killing own cells means killing own power 6)
            elif score == 0:
                check = 0
                # just a power 1 cell moving around
                if power == 1:
                    dir = action.direction
                    dir = (int(dir.r), int(dir.q))
                    tmp_coord = (from_cell[0] + dir[0], from_cell[1] + dir[1])
                    tmp_coord = self.fix_tuple(tmp_coord)

                    # if there are enemy around the cell right now and moving towards a safe cell, performance is a little wierd. When simulations
                    # are run, it seems like cells just spread randomly, if initial position is dangerous, and safter after moving   
                    if self.check_enemy(from_cell, colour) and not self.check_enemy(tmp_coord, colour):
                        average.append(action)
                        check = 1

                if not check:
                    bad.append(action)

            # Negative score (Spreading creates greater power nodes)
            else:
                tmp_colour_power = copy.colour_total_power(colour)
                check = 0
                # spreading on itself and not eating any enemy cells
                dir = action.direction
                dir = (int(dir.r), int(dir.q))
                for i in range(power):
                    spread_coord = self.add_tuple(from_cell, self.mult_tuple(dir, i + 1))
                    spread_coord = self.fix_tuple(spread_coord)
                    # if it spreads near enemy cells
                    if copy.check_enemy(spread_coord, colour):
                        bad.append(action)
                        check = 1
                        break
                    
                if not check and tmp_colour_power > 6 and tmp_colour_power == self.colour_total_power(colour) and num_own_colour > 4:
                    good.append(action)
                else:
                    bad.append(action)

            # undo action
            copy.undo_action(changes_dict)
        del copy

        # return the best action
        if len(good) != 0:
            return random.choice(good)
        elif len(average) != 0:
            return random.choice(average)
        else:
            if len(bad) != 0:
                return random.choice(bad)

    # Function that returns a guided random action (light weight compared to the other heuristic) 
    # While turns < 5, we are gonna lay greedy
    def light_heuristic(self, actions):
        good = []
        average = []
        bad = []
        
        # Not gonna use deepcopy for simulations since too expensive and doesn't really affect main branch of the tree

        for action in actions:

            # Initialize score and player_turn colour
            colour = self.player_turn
            score = 0
       
            # Get changes dict, and initial and new number of red and blue cells
            init_red = self.num_red
            init_blue = self.num_blue
            
            # Initial total power of the player_turn colour
            initial_colour_total_power = self.colour_total_power(colour)

            initial_total_power = self.total_power

            changes_dict = self.apply_action(action, action_param = "get_actions_dict")
            
            new_red = self.num_red
            new_blue = self.num_blue

            if colour == 'r':
                score = (new_red - init_red) + (init_blue - new_blue)
            else:
                score = (new_blue - init_blue) + (init_red - new_red)
           
            # Rank different actions in different lists
 
            # Positive: Spread action or spawn action that does more good than harm 
            if score > 0:
                flag = 0
                
                # Spawn action case: either random, next to enemy, or near own cell
                if isinstance(action, SpawnAction):
                    from_cell = action.cell
                    coordinates = (int(from_cell.r), int(from_cell.q))

                    # checking each of the neighbour cells
                    # make sure we're not spawning right next to an enemy cell
                    for dir in DIR.coord:
                        tmp_coord = (coordinates[0] + dir[0], coordinates[1] + dir[1])
                        tmp_coord = self.fix_tuple(tmp_coord)
                        if tmp_coord in self.grid_state:
                            # neighbour is an enemy
                            if self.eval_colour(tmp_coord) != colour:
                                bad.append(action)
                                flag = 1
                                break
                            # neighbour is own 
                            else:
                                flag = 2
                        if flag == 1:
                            break

                    # spawning in an empty surrounding
                    if flag == 0:
                        average.append(action)
                    # spawning next to own cell, with no enemy surrounding
                    elif flag == 2:
                        good.append(action)
	
######## Don't knwo whether this version is better or not
#                    if flag == 0:
#                        if initial_total_power > 20:
#                           bad.append(action)
#                        else:
#                            average.append(action)
#                    # spawning next to own cell, with no enemy surrounding
#                    elif flag == 2:
#                        if initial_total_power > 20:
#                            average.append(action)
#                        else:
#                            good.append(action)

                # Spread is good if it kills enemy nodes, bad if not
                elif isinstance(action, SpreadAction):
                    # Initialize 
                    cell, dir = action.cell, action.direction
                    from_cell = (int(cell.r), int(cell.q))
                    dir = (int(dir.r), int(dir.q))

                    # Spreading is good if it kills enemy nodes, otherwise it is bad
                    if colour == 'r':
                        if new_blue - init_blue == 0:
                            bad.append(action)
                        else:
                            if self.turns < 5: # Avoid spawning next to enemies (lose automatically for start of game)
                                self.undo_action(changes_dict)
                                return action
                            else:
                                good.append(action)
                    else:
                        if new_red - init_red == 0:
                            bad.append(action)
                        else:
                            if self.turns < 5: 
                                self.undo_action(changes_dict)
                                return action
                            else:
                                good.append(action)

            # Score is 0, Spread does more harm than good (kill own power 6 cell)
            elif score == 0:
                bad.append(action)

            else: 
                # Good action if total power of colour now is at least as large than it previous was
                if initial_colour_total_power <= self.colour_total_power(colour) and self.turns > 10:
                    good.append(action)
                else:
                    bad.append(action)
            
            # undo action to board (same time and memory)
            self.undo_action(changes_dict)

        
        if len(good) != 0:
            return random.choice(good)
        
        elif len(average) != 0:
            return random.choice(average)
        
        elif len(bad) != 0:
                return random.choice(bad)
        else:
            raise ValueError("This shouldn't run, the actions should have been in one of the lists")


    # Assuming coordinate is inside the board, returns the colour of the coordinate on the board
    def eval_colour(self, coordinate):
        return self.grid_state[coordinate][0]
    
    # Assuming coordinate is inside the board, returns the power of the coordinate on the board, returns power = 0 if it's not in grid
    def eval_power(self, coordinate):
        if coordinate in self.grid_state:
            return self.grid_state[coordinate][1]
        return 0

    # Function that does vector addition for 2 coordinates
    def add_tuple(self, a, b):
        return (a[0] + b[0], a[1] + b[1])

    # Function that does scalar multiplication on the tuple a
    def mult_tuple(self, a, scalar):
        return (scalar * a[0], scalar * a[1])
    
    # Function that "fixes" the tuple so that the tuple remains in the torus structure
    def fix_tuple(self, a):
        return (a[0] % 7, a[1] % 7)

    
    '''
    O(1), just accessing stuff
    '''
    @property  #@property means that you don't need () at the end of the method if you're not taking any parameters I think
    # Function that evalutes the board turns and returns the player colour that is to play in the current turn (Red: even, Blue: odd)
    def player_turn(self) -> str:
        # Red's turn when total turns is even
        if self.turns % 2 == 0:
            return 'r'
        else: #'b' plays on odd turn
            return 'b'
   

    # Function that takes a grid_state (dictionary) as input and outputs True (game has ended) or False (game hasn't ended) 
    '''
    O(1), just accessing stuff
    '''
    @property
    def game_over(self):
        if self.turns < 2: 
            return False

        return any([
            self.turns >= MAX_TURNS,
            self.num_red == 0,
            self.num_blue == 0
        ])

    # Function that board class as input, and outputs True (winning colour quite obvious) or False (winner colour hard to determine)
    @property
    def too_much_adv(self):
        # return True if a particular colour has at least 10 more nodes or power than the other colour (pretty evident wo will win)
        return any([
            abs(self.num_red - self.num_blue) >= 10,
            abs(self.colour_total_power('r') - self.colour_total_power('b')) >= 10 # This has searching whole dictionary complexity
        ])
    

    # Function that takes in a grid_state (dictionary) as input and outputs the colour of the winner ("R" or "B") 
    # Make sure that this function is only run after game_end condition is satisfied. There are only 4 conditions of end_game that determine winner
    '''
    O(1), just accessing stuff and comparing them
    '''
    @property
    def winner(self):
        # If board reached max number of turns (343 turns), the winner is the colour with the most power 
        if self.turns >= MAX_TURNS or not self.game_over:
            return self.max_power_colour()
        # Otherwise, empty board represents draw and the colour without nodes on the board is the LOSING colour 
        else:
            if (self.num_red == 0 and self.num_blue == 0):
                return None # need a colour here that represents draw, perhaps use "W" as a global constant ("WHITE"), neither blue or red has won
            elif self.num_red == 0:
                return 'b'
            else: # self.num_blue == 0
                if self.num_blue != 0:
                    raise ValueError("Something is wrong, perhaps the ending condition is not satisfied")
                return 'r'


    # Function that returns the player / colour with the most total power on the board 
    def max_power_colour(self):
        blue = 0
        red = 0
        for info in self.grid_state.values():
            if info[0] == 'r':
                red += info[1]
            else:
                blue += info[1]
        
        if red > blue:
            return 'r'
        # winner is blue if red <= blue for now
        else:
            return 'b'
        
    # Function that returns the total power of the colour (input)
    def colour_total_power(self, colour):
        count = 0 # Tally
        
        # Count colours
        for info in self.grid_state.values():
            if info[0] == colour:
                count += info[1]
        return count
    
    # function that returns the cell coordinate with the largest power, given player colour
    def largest_power_cell(self, colour):
        largest_power = 1 # will only consider as large power if power is > 1
        largest_cells = []
        for key, value in self.grid_state.items():
            if value[0] == colour:
                # if found a larger power cell, clear the largest cell list
                if value[1] > largest_power:
                    largest_power = value[1]
                    largest_cells.clear()
                    largest_cells.append(key)
                elif value[1] == largest_power and len(largest_cells) != 0:
                    largest_cells.append(key)

        return largest_cells



    # Function used for debugging purposes: prints the fields / attributes of the BOARD CLASS
    @property
    def print_board_data(self):
        print("\nPrinting Board Data:")
        print(f"The grid state is {self.grid_state}\n The board looks like this:")
        print(render_board(self.grid_state, ansi = True))
        print(f"The number of blue nodes on the board is {self.num_blue}")
        print(f"The number of red nodes on the board is {self.num_red}")
        print(f"The total power of the board is {self.total_power}")
        print(f"The number of turns of the board is {self.turns}\n")


# Class representing the Monte Carlo Tree
class MCT:
    def __init__(self, root: NODE): # Include time_remaining here possibly, make the number of iterations dependent on time, maybe current number of turns?
        self.root = root
        self.root.total = len(self.root.board.get_legal_actions)


    # perform monte carlo tree search: Initialize, Select, Expand, Simulate, Backpropogate
    def mcts(self):
        count = 0
        root = self.root
        
        while count < MAX_ITERATIONS: # Can include memory and time constraint in the while loop as well 

            # Traverse tree and select best node based on UCB until reach a node that isn't fully explored
            node = root
#            random = 1
            while not node.board.game_over and node.explored_enough:
#                print(f"tree traversed {random} times, it looks like this:")
#                print(render_board(node.board.grid_state, ansi = True))
#                random += 1
                node = node.largest_ucb()

           
            # Expansion: Expand if board not at terminal state and node still has unexplored children
            if not node.board.game_over and not node.explored_enough:
#                print(f"I was here, node is: {node}")
#                print(render_board(node.board.grid_state, ansi = True))
#                node.print_node_data
                node = node.expand() # Generates a child node that is most likely
#                print(f"afterwards, node is: {node}")

            # Simulation: Simulate newly expanded node or save winner of  the terminal state
            winner = node.simulate()

            # Backpropogation: Traverse to the root of the tree and update wins and playouts
            node.backpropogate(winner)

            count += 1

        action = root.best_final_action()
        return action

