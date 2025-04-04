# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent

from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir

from .search_strategy import * # Note written yet 

# Planning to write a function in a different python file and just import it into this afterwards 
MAX_ITERATIONS = 100


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
        self.mct = MCT(NODE(BOARD({}), total = 49))
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
                return self.mct.mcts(MAX_ITERATIONS)
                # return SpawnAction(HexPos(3, 3))
            case PlayerColor.BLUE:
                return self.mct.mcts(MAX_ITERATIONS)
                # This is going to be invalid... BLUE never spawned!
                #return SpreadAction(HexPos(3, 3), HexDir.Up)

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """

        # For each action, we need to update change the root of our MCTS tree
        flag = 0
        match action:
            case SpawnAction(cell):
                print(f"Testing: {color} SPAWN at {cell}")
                pass
            case SpreadAction(cell, direction):
                print(f"Testing: {color} SPREAD from {cell}, {direction}")
                pass

        for child in self.mct.root.children:
            # same action as child, set root as child
            if child.action == action:
                del self.mct.root.children
                self.mct.root = child
                flag = 1
                break

        
        if flag == 0:
            raise ValueError("Action not found in children")