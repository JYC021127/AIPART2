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
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as red")
                self.mct = MCT(NODE(BOARD({})))
                self._color = color
            case PlayerColor.BLUE:
                print("Testing: I am playing as blue")
                self.mct = MCT(NODE(BOARD({})))
                self._color = color

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """
        match self._color:
            case PlayerColor.RED:
                print("RED ACTION")
                return self.mct.mcts(time_left = referee['time_remaining'], space_left = referee["space_limit"])

            case PlayerColor.BLUE:
                print("BLUE ACTION")
                return self.mct.mcts(time_left = referee['time_remaining'], space_left = referee["space_limit"])

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
        #print(f"Time remaining:  {referee['time_remaining']}")
        #print(f"Space remaining: {referee['space_remaining']}\n")
        self.update(action)


    # Function that updates root of the tree based on enemy action  
    def update(self, action: Action):
        flag = 1
        for child in self.mct.root.children:
            # same action as child, set root as child
            if child.action == action:
                # Rely on inbuild python memory recycling
#                del self.mct.root.children
                child.parent = None
                child.action = None
                self.mct.root = child
                flag = 0
                #print("action found...")
                break
        
        # Opponent's move is not found in root.children
        if flag:
            previous = self.mct.root # previous root 
            previous.board.apply_action(action)  # modify root since not using it anymore

            tmp = previous.board.get_legal_actions

            self.mct.root = NODE(board = deepcopy(previous.board), actions_left = tmp, total = len(tmp))
            del previous


