# This was coded as part of an assignment that used the Pacman AI projects from UC Berkeley (http://ai.berkeley.edu)
# However, as these projects were not developed by me, all game files were omitted except for this one. The bare minimum
# of the code (mostly class declarations and comments) was kept so that the algorithms are shown properly implemented.


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

# Added import of maxint for "positive infinity" (biggest_number) and "negative infinity" (smallest_number)
from sys import maxint
biggest_number = maxint
smallest_number = (-1)*biggest_number


# Helper Functions for distances in lists
def min_ghost_from_list(ref_post, ghost_list):
    min_ret = biggest_number
    ghost_scared = False
    for ghost in ghost_list:
        curr_dist = util.manhattanDistance(ref_post, ghost.getPosition())
        if curr_dist < min_ret:
            min_ret = curr_dist
            ghost_scared = (ghost.scaredTimer > 0)
    return min_ret, ghost_scared


def min_max_dist_from_list(ref_pos, pos_list):
    min_ret, max_ret = biggest_number, 0
    for element in pos_list:
        curr_dist = util.manhattanDistance(ref_pos, element)
        if curr_dist < min_ret:
            min_ret = curr_dist
        elif curr_dist > max_ret:
            max_ret = curr_dist
    return min_ret, max_ret


# Original ReflexAgent Class
class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Start with the provided base line for the score and define a base point value to avoid a few magic numbers
        current_score = successorGameState.getScore()
        base_pts = 10

        # First check for a win, if so we found a good state and can exit
        if successorGameState.isWin():
            return current_score + 1000000

        # Otherwise check through main categories: Food, Ghosts, Capsules
        # Food Related
        # Add points if we ate some food (there's probably a less crude way of doing this)
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            current_score += 2*base_pts

        # Find min, max dist to food and add (weighted) to score (convert newFood into list of pairs and pass to helper)
        min_dist_to_food, max_dist_to_food = min_max_dist_from_list(newPos, newFood.asList())
        current_score -= min_dist_to_food
        current_score -= 0.01*max_dist_to_food

        # Ghost Related
        # Add weighted min dist to score (note inversion check), if ghost gets close handle with special cases
        min_ghost_dist, ghost_invert = min_ghost_from_list(newPos, newGhostStates)
        if min_ghost_dist == 2:
            current_score -= ((-1)*base_pts) if ghost_invert else base_pts
        elif min_ghost_dist == 1:
            current_score -= 10*(((-1)*base_pts) if ghost_invert else base_pts)
        elif min_ghost_dist == 0:
            current_score -= 50*(((-1)*base_pts) if ghost_invert else base_pts)
        else:
            current_score += 0.02*min_ghost_dist

        # "Power Pellet" / Capsule Related
        # Add points if we ate a capsule (there's probably a less crude way of doing this)
        if currentGameState.getCapsules() > successorGameState.getCapsules():
            current_score += 8*base_pts

        # End of scoring
        # For debugging: print action + " " + str((current_score-currentGameState.getScore()))
        return current_score


def scoreEvaluationFunction(currentGameState):
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


# Added helper functions for minimax/alpha-beta agents
def is_end_state(game_state):
    return game_state.isWin() or game_state.isLose()


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        # Maximize Pacman, Minimize Ghosts
        # Referred to Slide 38 from Game Tree Search Lecture for general psuedocode/approach

        # The maximizer/minimizer portions were split off to make things easier as there are different options available
        # for Pacman and handling the multiple ghost case is a little tricky (and thus easier to address in isolation)

        # Identify the number of ghosts for use later
        total_ghosts = gameState.getNumAgents() - 1  # -1 for Pacman

        # Define maximizer
        def maximizer(game_state, depth):
            # Terminal state check
            if is_end_state(game_state) or depth == self.depth:
                return self.evaluationFunction(game_state)

            # Get actions and call the minimizer for each possible action, returning the best one
            current_max = smallest_number
            for act in game_state.getLegalActions():
                current_max = max(current_max, minimizer(game_state.generateSuccessor(0, act), depth, 1))
            return current_max

        # Define minimizer - but there are multiple min-agents/ghosts which complicates things
        # Let ghost number be the agent number of the ghost (could view it as an index 1,2,...,n of the list of ghosts)
        def minimizer(game_state, depth, ghost_number):
            # Terminal state check
            if is_end_state(game_state) or depth == self.depth:
                return self.evaluationFunction(game_state)

            # Get actions BUT we must minimize over all the ghosts at the current depth so we should call minimizer
            # for every single action over every single agent at this depth, then call the maximizer when we're done
            # This approach still works if there is one ghost as it will immediately jump to elif case
            # Note: The conditionals (i.e. if/elif/else) use total_ghosts, a var set above
            current_min = biggest_number
            for act in game_state.getLegalActions(ghost_number):
                if ghost_number < total_ghosts:
                    current_min = min(current_min, minimizer(game_state.generateSuccessor(ghost_number, act), depth, ghost_number+1))
                elif ghost_number == total_ghosts:
                    current_min = min(current_min, maximizer(game_state.generateSuccessor(ghost_number, act), depth+1))
                else:
                    raise Exception("Minimizer called on too many ghosts!")  # Should never have ghost_number > total_ghosts

            return current_min

        # DFMiniMax Implementation - special case for Level 0 as we care about the max value action not the max value
        def maximizer_L0(game_state, depth):
            # Terminal state check
            if is_end_state(game_state) or depth == self.depth:
                return self.evaluationFunction(game_state)

            # Get actions and call the minimizer for each possible action
            # Start with base case of smallest number and first action
            possible_actions = game_state.getLegalActions()
            current_max = (smallest_number, possible_actions[0])
            for act in possible_actions:
                current_minimizer_output = minimizer(game_state.generateSuccessor(0, act), depth, 1)
                if current_max[0] < current_minimizer_output:
                    current_max = (current_minimizer_output, act)

            # Get a directly ref to best action and return it (so we're not returning part of a tuple)
            ret_action = current_max[1]
            return ret_action

        # We are Pacman and can just call the L0 maximizer directly and it will return the best action
        return maximizer_L0(gameState, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # This is the exact same code as above (i.e. initially copied) except for the addition of Alpha-Beta pruning

        # The maximizer/minimizer portions were split off to make things easier as there are different options available
        # for Pacman and handling the multiple ghost case is a little tricky (and thus easier to address in isolation)

        # Identify the number of ghosts for use later
        total_ghosts = gameState.getNumAgents() - 1  # -1 for Pacman

        # Define maximizer
        def ab_maximizer(game_state, depth, alpha, beta):
            # Terminal state check
            if is_end_state(game_state) or depth == self.depth:
                return self.evaluationFunction(game_state)

            # Get actions and call the minimizer for each possible action, returning the best one
            current_max = smallest_number
            for act in game_state.getLegalActions():
                current_max = max(current_max, ab_minimizer(game_state.generateSuccessor(0, act), depth, 1, alpha, beta))
                # AlphaBeta addition
                # Update alpha since at a max node alpha is max value found among children of current state
                # After update check if A >= B. If so, we can stop and return the value. (i.e. prune)
                alpha = max(alpha, current_max)
                if alpha >= beta:
                    return alpha

            return current_max

        # Define minimizer - but there are multiple min-agents/ghosts which complicates things
        # Let ghost number be the agent number of the ghost (could view it as an index 1,2,...,n of the list of ghosts)
        def ab_minimizer(game_state, depth, ghost_number, alpha, beta):
            # Terminal state check
            if is_end_state(game_state) or depth == self.depth:
                return self.evaluationFunction(game_state)

            # Get actions BUT we must minimize over all the ghosts at the current depth so we should call minimizer
            # for every single action over every single agent at this depth, then call the maximizer when we're done
            # This approach still works if there is one ghost as it will immediately jump to elif case
            # Note: The conditionals (i.e. if/elif/else) use total_ghosts, a var set above
            current_min = biggest_number
            for act in game_state.getLegalActions(ghost_number):
                if ghost_number < total_ghosts:
                    current_min = min(current_min, ab_minimizer(game_state.generateSuccessor(ghost_number, act), depth, ghost_number+1, alpha, beta))
                elif ghost_number == total_ghosts:
                    current_min = min(current_min, ab_maximizer(game_state.generateSuccessor(ghost_number, act), depth+1, alpha, beta))
                else:
                    raise Exception("Minimizer called on too many ghosts!")  # Should never have ghost_number > total_ghosts
                # AlphaBeta addition
                # Update beta since at a min node beta is min value found among children of current state
                # After update check if A >= B. If so, we can stop and return the value. (i.e. prune)
                beta = min(beta, current_min)
                if alpha >= beta:
                    return beta

            return current_min

        # AlphaBeta Implementation - special case for Level 0 as we care about the max value action not the max value plus we don't have to check alpha >= beta
        def ab_maximizer_L0(game_state, depth):
            # Terminal state check
            if is_end_state(game_state) or depth == self.depth:
                return self.evaluationFunction(game_state)

            # Get actions and call the minimizer for each possible action
            # Start with base case of smallest number and first action, plus alpha=-inf, beta=inf
            possible_actions = game_state.getLegalActions()
            current_max = (smallest_number, possible_actions[0])
            alpha, beta = smallest_number, biggest_number
            for act in possible_actions:
                current_minimizer_output = ab_minimizer(game_state.generateSuccessor(0, act), depth, 1, alpha, beta)
                if current_max[0] < current_minimizer_output:
                    current_max = (current_minimizer_output, act)
                # AlphaBeta addition
                # Update alpha since at a max node alpha is max value found among children of current state
                # Beta will be irrelevant since this is L0 (beta will always be positive infinity)
                alpha = max(alpha, current_max[0])

            # Get a directly ref to best action and return it (so we're not returning part of a tuple)
            ret_action = current_max[1]
            return ret_action

        # We are Pacman and can just call the L0 maximizer directly and it will return the best action
        return ab_maximizer_L0(gameState, 0)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # This is the exact same code as above (i.e. initially copied) except for the conversion to Expectimax

        # The maximizer/minimizer portions were split off to make things easier as there are different options available
        # for Pacman and handling the multiple ghost case is a little tricky (and thus easier to address in isolation)

        # Identify the number of ghosts for use later
        total_ghosts = gameState.getNumAgents() - 1  # -1 for Pacman

        # Define maximizer
        def maximizer(game_state, depth):
            # Terminal state check
            if is_end_state(game_state) or depth == self.depth:
                return self.evaluationFunction(game_state)

            # Get actions and call the minimizer for each possible action, returning the best one
            current_max = smallest_number
            for act in game_state.getLegalActions():
                current_max = max(current_max, expecti_minimizer(game_state.generateSuccessor(0, act), depth, 1))
            return current_max

        # Define minimizer - but there are multiple min-agents/ghosts which complicates things
        # Let ghost number be the agent number of the ghost (could view it as an index 1,2,...,n of the list of ghosts)
        def expecti_minimizer(game_state, depth, ghost_number):
            # Terminal state check
            if is_end_state(game_state) or depth == self.depth:
                return self.evaluationFunction(game_state)

            # Get actions BUT we must minimize over all the ghosts at the current depth so we should call minimizer
            # for every single action over every single agent at this depth, then call the maximizer when we're done
            # This approach still works if there is one ghost as it will immediately jump to elif case
            # Note: The conditionals (i.e. if/elif/else) use total_ghosts, a var set above
            current_avg = 0.0
            for act in game_state.getLegalActions(ghost_number):
                if ghost_number < total_ghosts:
                    current_avg += float(expecti_minimizer(game_state.generateSuccessor(ghost_number, act), depth, ghost_number+1))
                elif ghost_number == total_ghosts:
                    current_avg += float(maximizer(game_state.generateSuccessor(ghost_number, act), depth+1))
                    current_avg /= float(total_ghosts)
                else:
                    raise Exception("Minimizer called on too many ghosts!")  # Should never have ghost_number > total_ghosts

            return current_avg

        # Expectimax Implementation - special case for Level 0 as we care about the max value action not the max value
        def maximizer_L0(game_state, depth):
            # Terminal state check
            if is_end_state(game_state) or depth == self.depth:
                return self.evaluationFunction(game_state)

            # Get actions and call the minimizer for each possible action
            # Start with base case of smallest number and first action
            possible_actions = game_state.getLegalActions()
            current_max = (smallest_number, possible_actions[0])
            for act in possible_actions:
                current_minimizer_output = expecti_minimizer(game_state.generateSuccessor(0, act), depth, 1)
                if current_max[0] < current_minimizer_output:
                    current_max = (current_minimizer_output, act)

            # Get a directly ref to best action and return it (so we're not returning part of a tuple)
            ret_action = current_max[1]
            return ret_action

        # We are Pacman and can just call the L0 maximizer directly and it will return the best action
        return maximizer_L0(gameState, 0)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    curr_pos = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()

    # Start with the provided base line for the score
    current_score = currentGameState.getScore()

    # Define category-level weight factors
    food_weight_factor = 2
    ghost_weight_factor = 4
    capsule_weight_factor = 10

    # First check for a win, if so we found a good state and can exit
    if currentGameState.isWin():
        return current_score + 1000000

    # Otherwise check through main categories: Food, Ghosts, Capsules
    # Food Related
    # Find min, max dist to food and add (weighted) to score (convert newFood into list of pairs and pass to helper)
    # min_dist_to_food, max_dist_to_food = min_max_dist_from_list(curr_pos, currentGameState.getFood().asList())

    # Use the advice from the lab handout and add the inverse
    min_dist_to_food = biggest_number
    food_pts = 0
    for food in currentGameState.getFood().asList():
        curr_dist = util.manhattanDistance(curr_pos, food)
        min_dist_to_food = min(min_dist_to_food, curr_dist)
        food_pts = food_weight_factor/curr_dist

    current_score += food_pts + (4*food_weight_factor/min_dist_to_food)

    # Ghost Related
    # Add weighted scores from ghosts to the current score
    # We may want to weight this more heavily i.e. by going from 2nd hyperop (mult/div) to 3rd (exp/log)
    min_ghost_dist = biggest_number
    for ghost in ghosts:
        curr_dist = util.manhattanDistance(curr_pos, ghost.getPosition())
        min_ghost_dist = min(curr_dist, min_ghost_dist)
        try:
            to_add = ghost_weight_factor/curr_dist
        except ZeroDivisionError:
            to_add = ghost_weight_factor*2
        current_score += to_add if (ghost.scaredTimer > 0) else ((-1) * to_add)

    # "Power Pellet" / Capsule Related
    # Add points if we ate a capsule (there's probably a less crude way of doing this)
    min_capsule_dist = biggest_number
    for caps in currentGameState.getCapsules():
        min_capsule_dist = min(min_capsule_dist, manhattanDistance(caps, curr_pos))

    # Use the advice from the lab handout and add the inverse
    current_score += capsule_weight_factor/min_capsule_dist

    # End of scoring
    # print "State: " + str(currentGameState.getPacmanPosition()) + " Score: " + str(current_score)
    return current_score


# Abbreviation
better = betterEvaluationFunction

