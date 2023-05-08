# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent

from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir

from .search_strategy import *

from copy import deepcopy


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
        self.mct = MCT(NODE(BOARD({})))
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
                return self.mct.mcts()

            case PlayerColor.BLUE:
                print("BLUE ACTION")
                return self.mct.mcts()

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

        self.update(action, color)
    


    def update(self, action: Action, color: PlayerColor):

        if color != self._color:
            print(f"Player is {color}, but updating {self._color} tree")


        flag = 1
        for child in self.mct.root.children:
            #child.print_node_data
            # same action as child, set root as child
            if child.action == action:
                # Rely on inbuild python memory recycling
#                del self.mct.root.children
                self.mct.root = child

                # print("root.children:")
                # print(self.mct.root.children)
                flag = 0
                print("\n\n\nAction found in some child node, tree updated \n")
                break
        
        # Opponent's move is not found in root.children
        if flag:
            print("\n\n\nAction not found in child node, new tree created \n")
            parent = self.mct.root
            new_board = deepcopy(self.mct.root.board)
            new_board.apply_action(action)
            new_child = NODE(board = new_board, parent = parent, action = action, total = len(self.mct.root.board.get_legal_actions))
            self.mct.root.children.append(new_child)
            self.mct.root = new_child
            print(f"new root is {self.mct.root}")


