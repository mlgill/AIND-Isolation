"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO)

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def L1_score_difference(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


def L2_score_difference(game, player):
    """An evaluation function that outputs a score equal to the difference 
    in the squares of number of moves available to the two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves**2 - opp_moves**2)


def L2_score_difference_and_location(game, player):
    """An evaluation function that outputs a score equal to the difference 
    in the squares of number of moves available to the two players and weights
    moves at the edge of the board more heavily.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    from math import fabs

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Find proximity to corner
    loc = game.get_player_location(player)

    board_middle = int(game.height/2)
    if game.height % 2:
        board_middle += 1

    loc = list(loc)
    loc[0] = fabs(board_middle - loc[0])
    loc[1] = fabs(board_middle - loc[1])
    loc = tuple(loc)

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves**2 - opp_moves**2 + loc[0] + loc[1])



def random_score_difference(game, player):
    """A function  that outputs a random score between 0 and 1.

    Parameters
    ----------
    none

    Returns
    ----------
    float
        The heuristic value of the current game state
    """

    return float(random.random() - 0.5)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return L2_score_difference_and_location(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):

        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left


        # Setup the search type
        if self.method == 'minimax':
            search_type = self.minimax
        else:
            search_type = self.alphabeta


        # Setup the initial best move
        best_move = (-1, -1)
        if len(legal_moves) > 0:
            best_move = legal_moves[0]


        # Setup the search range to either cover the entire range allowed by the system
        # or just that specified during initializaiton
        if self.iterative:
            search_range = (1, sys.maxsize)
        else:
            search_range = (self.search_depth, self.search_depth + 1)


        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            for depth in range(search_range[0], search_range[1]):

                best_score, best_move = search_type(game, depth)

                #logging.info('Timeout: {}'.format(tl))

                # Stop if at the bottom of the search tree
                if ((best_score == float("+inf")) or (best_score == float("-inf"))):
                    break

        except Timeout:
            # Handle any actions required at timeout, if necessary

            if best_move == (-1, -1):
                real_moves = game.get_legal_moves()
                if len(real_moves) > 0:
                    return real_moves[-1]
            pass

        return best_move


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Get a list of remaining moves for the current player
        remaining_moves = game.get_legal_moves()

        # The exit conditions for the recursion
        if len(remaining_moves) == 0:
            return self.score(game, self), (-1, -1)
        elif depth == 0:
            return self.score(game, self), remaining_moves[0]

            
        # Run minimax for each move recursively until the maximum depth has been reached
        move_and_score = [(self.minimax(game.forecast_move(m), 
                               depth - 1, 
                               np.invert(maximizing_player)), m) for m in remaining_moves]
        
        
        # Get max or min, depending on which player is active
        if maximizing_player:
            ((score, X), selected_move) = max(move_and_score)
        else:
            ((score, X), selected_move) = min(move_and_score)
        
        return score, selected_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Get a list of remaining moves for the current player
        remaining_moves = game.get_legal_moves()

        # The exit conditions for the recursion
        if len(remaining_moves) == 0:
            return self.score(game, self), (-1, -1)
        elif depth == 0:
            return self.score(game, self), remaining_moves[0]

            
        # Run alphabeta for each move recursively until the maximum depth has been reached
        move_and_score = list()
        for m in remaining_moves:
            score, X = self.alphabeta(game.forecast_move(m), 
                                      depth - 1, 
                                      alpha, 
                                      beta, 
                                      np.invert(maximizing_player))
        
            #logging.info('{} {}'.format(score, X))
            move_and_score.append((score, m))

            if (maximizing_player and (score > alpha)):
                    alpha = score
            elif ((not maximizing_player) and (score < beta)): 
                    beta = score

            if beta <= alpha:
                # logging.info('Pruned a leaf.')
                break

        # Get max or min, depending on which player is active
        if maximizing_player:
            final_score = max(move_and_score)
        else:
            final_score = min(move_and_score)
        
        return final_score
        
