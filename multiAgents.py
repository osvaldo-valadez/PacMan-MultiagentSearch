# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from statistics import mean

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print(f"The successor Game state is {successorGameState}")
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodList = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        ghostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        maxScareTime = max(newScaredTimes)
        capsules = successorGameState.getCapsules()

        score = successorGameState.getScore()
        
        if len(newFoodList) > 0:
            closestFood = min(newFoodList, key = lambda x: manhattan(x, newPos))
            score += 2* (1/(manhattan(closestFood, newPos)))
        if len(successorGameState.getCapsules()) < len(currentGameState.getCapsules()):
            score+= 500
        if len(successorGameState.getCapsules()) > 0:
            closestCapsule = min(capsules, key = lambda x: manhattan(x, newPos))
            score += (1 / manhattan(closestCapsule, newPos))
        if len(ghostPositions) != 0 and maxScareTime == 0:
            closestGhost = min(ghostPositions, key = lambda x: manhattan(x, newPos))
            howClose = manhattan(closestGhost, newPos)
            if howClose > 0:
                score -= (1/howClose)
        if successorGameState.hasWall(newPos[0],newPos[1]):
            return 0
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        numAgents = gameState.getNumAgents()
        def maxAgent(gameState, depth):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            return max([minAgent(gameState.generateSuccessor(0, action), depth, 1) for action in gameState.getLegalActions(0)])
        def minAgent(gameState, depth, ghostNumber):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if ghostNumber == numAgents - 1:
                return min([maxAgent(gameState.generateSuccessor(ghostNumber, action), depth - 1) for action in gameState.getLegalActions(ghostNumber)])
            else: 
                return min([minAgent(gameState.generateSuccessor(ghostNumber, action), depth, ghostNumber + 1) for action in gameState.getLegalActions(ghostNumber)])
            
        maxScore = maxAgent(gameState, self.depth)
        for action in gameState.getLegalActions(0):
            if minAgent(gameState.generateSuccessor(0, action), self.depth, 1) == maxScore:
                return action
            
            
        
                

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()
        def maxAgent(gameState, depth, alpha, beta):
            v = float('-inf')
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            for action in gameState.getLegalActions(0):
                v = max([v, minAgent(gameState.generateSuccessor(0, action), depth, 1, alpha, beta)])
                if v > beta:
                    return v
                alpha = max([alpha, v])
            return v
            
            
        def minAgent(gameState, depth, ghostNumber, alpha, beta):
            v = float('inf')
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            for action in gameState.getLegalActions(ghostNumber): 
                if ghostNumber == numAgents - 1:
                    v = min([v, maxAgent(gameState.generateSuccessor(ghostNumber, action), depth - 1, alpha, beta)])
                else:
                    v = min([v, minAgent(gameState.generateSuccessor(ghostNumber, action), depth, ghostNumber + 1, alpha, beta)])
                if v < alpha:
                    return v
                beta = min([beta, v])
            return v
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        v = float('-inf')
        for action in gameState.getLegalActions(0):
            value = minAgent(gameState.generateSuccessor(0, action), self.depth, 1, alpha, beta)
            if v < value:
                v = value
                bestAction = action
            if v > beta:
                return bestAction
            alpha = max([alpha, value])
        return bestAction
                    
            
            
        
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        numAgents = gameState.getNumAgents()
        def maxAgent(gameState, depth):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            return max([expectAgent(gameState.generateSuccessor(0, action), depth, 1) for action in gameState.getLegalActions(0)])
        def expectAgent(gameState, depth, ghostNumber):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if ghostNumber == numAgents - 1:
                return mean([maxAgent(gameState.generateSuccessor(ghostNumber, action), depth - 1) for action in gameState.getLegalActions(ghostNumber)])
            else: 
                return mean([expectAgent(gameState.generateSuccessor(ghostNumber, action), depth, ghostNumber + 1) for action in gameState.getLegalActions(ghostNumber)])
            
        maxScore = maxAgent(gameState, self.depth)
        for action in gameState.getLegalActions(0):
            if expectAgent(gameState.generateSuccessor(0, action), self.depth, 1) == maxScore:
                return action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    newGhostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    maxScareTime = max(newScaredTimes)
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()
    
    if len(newFoodList) > 0:
        closestFood = min(newFoodList, key = lambda x: manhattan(x, newPos))
        score += 2* (1/(manhattan(closestFood, newPos)))
    if len(currentGameState.getCapsules()) > 0:
        closestCapsule = min(capsules, key = lambda x: manhattan(x, newPos))
        score += (1 / manhattan(closestCapsule, newPos))
    if len(ghostPositions) != 0 and maxScareTime == 0:
        closestGhost = min(ghostPositions, key = lambda x: manhattan(x, newPos))
        howClose = manhattan(closestGhost, newPos)
        if howClose > 0:
            score -= (1/howClose)

    return score

def manhattan(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
# Abbreviation
better = betterEvaluationFunction
