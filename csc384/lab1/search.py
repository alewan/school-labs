# This was coded as part of an assignment that used the Pacman AI projects from UC Berkeley (http://ai.berkeley.edu)
# However, as these projects were not developed by me, all game files were omitted except for this one. The bare minimum
# of the code (mostly class declarations and comments) was kept so that the algorithms are shown properly implemented.


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # Declare data structures
    frontier = util.Stack()

    # Attempt to use a set first for efficiency, if it doesn't work then use a list
    try:
        visited = set()

        # Get initial state and add it to the frontier
        frontier.push((problem.getStartState(), []))

        # Expand frontier
        while not frontier.isEmpty():
            # Get current S,A pair
            c_state, c_actions = frontier.pop()

            # Check if it is a goal state, if so return the actions
            if problem.isGoalState(c_state):
                return c_actions

            # Expand all successors of current state and add them to the frontier if not already visited
            visited.add(c_state)
            for successor in problem.getSuccessors(c_state):
                if not (successor[0] in visited):
                    temp_l = [successor[1]]
                    frontier.push((successor[0], c_actions + temp_l))
    except TypeError:
        visited = list()

        # Get initial state and add it to the frontier
        frontier.push((problem.getStartState(), []))

        # Expand frontier
        while not frontier.isEmpty():
            # Get current S,A pair
            c_state, c_actions = frontier.pop()

            # Check if it is a goal state, if so return the actions
            if problem.isGoalState(c_state):
                return c_actions

            # Expand all successors of current state and add them to the frontier if not already visited
            visited.append(c_state)
            for successor in problem.getSuccessors(c_state):
                if not (successor[0] in visited):
                    temp_l = [successor[1]]
                    frontier.push((successor[0], c_actions + temp_l))

    # If not found by this point, it doesn't exist (just return empty list as dummy)
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #Declare data structures
    frontier = util.Queue()

    # Attempt to use a set first for efficiency, if it doesn't work then use a list
    try:
        visited = set()

        # Get initial state and add it to the frontier
        frontier.push((problem.getStartState(), []))
        visited.add(problem.getStartState())

        # Expand frontier
        while not frontier.isEmpty():
            # Get current S,A pair
            c_state, c_actions = frontier.pop()

            # Check if it is a goal state, if so return the actions
            if problem.isGoalState(c_state):
                return c_actions

            # Expand all successors of current state and add them to the frontier if not already visited
            for successor in problem.getSuccessors(c_state):
                if not (successor[0] in visited):
                    temp_l = [successor[1]]
                    frontier.push((successor[0], c_actions + temp_l))
                    visited.add(successor[0])
    except TypeError:
        visited = list()

        # Get initial state and add it to the frontier
        frontier.push((problem.getStartState(), []))
        visited.append(problem.getStartState())

        # Expand frontier
        while not frontier.isEmpty():
            # Get current S,A pair
            c_state, c_actions = frontier.pop()

            # Check if it is a goal state, if so return the actions
            if problem.isGoalState(c_state):
                return c_actions

            # Expand all successors of current state and add them to the frontier if not already visited
            for successor in problem.getSuccessors(c_state):
                if not (successor[0] in visited):
                    temp_l = [successor[1]]
                    frontier.push((successor[0], c_actions + temp_l))
                    visited.append(successor[0])

        # If not found by this point, it doesn't exist (just return empty list as dummy)
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Declare data structures
    frontier = util.PriorityQueue()

    # Try to use a dictionary first, if we can't then use a regular list
    try:
        visited = dict()

        # Get initial state and add it to the frontier and visited
        frontier.push((problem.getStartState(), []), 0)
        visited[problem.getStartState()] = 0

        # Expand frontier
        while not frontier.isEmpty():
            # Get current S,A pair
            c_sa = frontier.pop()
            c_state = c_sa[0]
            c_actions = c_sa[1]

            # Check if it is a goal state, if so return the actions
            if problem.isGoalState(c_state):
                return c_actions

            # Expand all successors of current state and add them to the frontier if not already visited
            for successor in problem.getSuccessors(c_state):
                temp_l = [successor[1]]
                temp_actions = c_actions + temp_l
                temp_cost_actions = problem.getCostOfActions(temp_actions)
                if (not (successor[0] in visited)) or (visited[successor[0]] > temp_cost_actions):
                    frontier.push((successor[0], temp_actions, temp_cost_actions), temp_cost_actions)
                    visited[successor[0]] = temp_cost_actions
    except TypeError:
        visited = list()

        # Get initial state and add it to the frontier and visited
        frontier.push((problem.getStartState(), []), 0)
        visited.append(problem.getStartState(), 0)

        # Expand frontier
        while not frontier.isEmpty():
            # Get current S,A pair
            c_sa = frontier.pop()
            c_state = c_sa[0]
            c_actions = c_sa[1]

            # Check if it is a goal state, if so return the actions
            if problem.isGoalState(c_state):
                return c_actions

            for successor in problem.getSuccessors(c_state):
                temp_l = [successor[1]]
                temp_actions = c_actions + temp_l
                temp_cost_actions = problem.getCostOfActions(temp_actions)
                temp_add = True

                for node in visited:
                    if successor[0] == node[0]:
                        temp_add = (node[1] > temp_cost_actions)
                        break

                if temp_add:
                    frontier.push((successor[0], temp_actions, temp_cost_actions), temp_cost_actions)
                    visited.append((successor[0], temp_cost_actions))

    # If not found by this point, it doesn't exist (just return empty list as dummy)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Declare data structures
    frontier = util.PriorityQueue()

    # Try to use a dictionary first, if we can't then use a regular list
    try:
        visited = dict()

        # Get initial state and add it to the frontier
        frontier.push((problem.getStartState(), []), heuristic(problem.getStartState(), problem))
        visited[problem.getStartState()] = heuristic(problem.getStartState(), problem)

        # Expand frontier
        while not frontier.isEmpty():
            # Get current S,A pair
            c_sa = frontier.pop()
            c_state = c_sa[0]
            c_actions = c_sa[1]

            # Check if it is a goal state, if so return the actions
            if problem.isGoalState(c_state):
                return c_actions

            # Expand all successors of current state and add them to the frontier if not already visited
            for successor in problem.getSuccessors(c_state):
                temp_l = [successor[1]]
                temp_actions = c_actions + temp_l
                temp_cost_actions = problem.getCostOfActions(temp_actions) + heuristic(successor[0], problem)
                if (not (successor[0] in visited)) or (visited[successor[0]] > temp_cost_actions):
                    frontier.push((successor[0], temp_actions, temp_cost_actions), temp_cost_actions)
                    visited[successor[0]] = temp_cost_actions
    except TypeError:
        visited = list()

        # Get initial state and add it to the frontier
        frontier.push((problem.getStartState(), []), heuristic(problem.getStartState(), problem))
        visited.append((problem.getStartState(), heuristic(problem.getStartState(), problem)))

        # Expand frontier
        while not frontier.isEmpty():
            # Get current S,A pair
            c_sa = frontier.pop()
            c_state = c_sa[0]
            c_actions = c_sa[1]

            # Check if it is a goal state, if so return the actions
            if problem.isGoalState(c_state):
                return c_actions

            # Expand all successors of current state and add them to the frontier if not already visited
            for successor in problem.getSuccessors(c_state):
                temp_l = [successor[1]]
                temp_actions = c_actions + temp_l
                temp_cost_actions = problem.getCostOfActions(temp_actions) + heuristic(successor[0], problem)
                temp_add = True

                for node in visited:
                    if successor[0] == node[0]:
                        temp_add = (node[1] > temp_cost_actions)
                        break

                if temp_add:
                    frontier.push((successor[0], temp_actions, temp_cost_actions), temp_cost_actions)
                    visited.append((successor[0], temp_cost_actions))
    # If not found by this point, it doesn't exist (just return empty list as dummy)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
