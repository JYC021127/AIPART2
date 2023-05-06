# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent

from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir

from .search_strategy import * # Note written yet 


from copy import deepcopy

# Planning to write a function in a different python file and just import it into this afterwards 
MAX_ITERATIONS = 1


# This is the entry point for your game playing agent. Currently the agent
# simply spawns a token at the centre of the board if playing as RED, and
# spreads a token at the centre of the board if playing as BLUE. This is
# intended to serve as an example of how to use the referee API -- obviously
# this is not a valid strategy for actually playing the game!

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initialise the agent.
        """
        self.mct = MCT(NODE(BOARD({}))) # This should be get legal moves instead of 49, not sure how to change this atm
        self._color = color

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as red")
            case PlayerColor.BLUE:
                print("Testing: I am playing as blue")

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """
        match self._color:
            case PlayerColor.RED:
                print("RED ACTION")
                return self.mct.mcts(MAX_ITERATIONS)
                # return SpawnAction(HexPos(3, 3))
            case PlayerColor.BLUE:
                print("BLUE ACTION")
                return self.mct.mcts(MAX_ITERATIONS)
                # This is going to be invalid... BLUE never spawned!
                #return SpreadAction(HexPos(3, 3), HexDir.Up)

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """

        # For each action, we need to update change the root of our MCTS tree
        match action:
            case SpawnAction(cell):
                print(f"Testing: {color} SPAWN at {cell}")
                pass
            case SpreadAction(cell, direction):
                print(f"Testing: {color} SPREAD from {cell}, {direction}")
                pass
        self.update(action)
        #self.mct.root.board.print_board_data
        # UPDATE:
            # change root node
            # if no children
                # apply action to current board (root node) -> new node


    def update(self, action: Action):
        flag = 0
        for child in self.mct.root.children:
            # same action as child, set root as child
            if child.action == action:
                print(child.children)
                del self.mct.root.children
                self.mct.root = child
                flag = 1
                break

#        if flag == 0:
#            previous = self.mct.root
#            new = previous.board.apply_action(action)
#            previous.print_node_data
#            previous.board.print_board_data
#            self.mct.root = NODE(new, total = len(new.get_legal_actions))
#            del previous

        if flag == 0:
            previous = self.mct.root #previous root 
#            print("previous board is:")
#            previous.board.print_board_data

            previous.board.apply_action(action)
            self.mct.root = NODE(board = previous.board, total = len(previous.board.get_legal_actions))
            del previous
