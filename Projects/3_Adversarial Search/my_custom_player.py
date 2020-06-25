import random
import time
import os
from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    # def get_action(self, state):
    #     """ Employ an adversarial search technique to choose an action
    #     available in the current state calls self.queue.put(ACTION) at least

    #     This method must call self.queue.put(ACTION) at least once, and may
    #     call it as many times as you want; the caller will be responsible
    #     for cutting off the function after the search time limit has expired.

    #     See RandomPlayer and GreedyPlayer in sample_players for more examples.

    #     **********************************************************************
    #     NOTE: 
    #     - The caller is responsible for cutting off search, so calling
    #       get_action() from your own code will create an infinite loop!
    #       Refer to (and use!) the Isolation.play() function to run games.
    #     **********************************************************************
    #     """
    #     # TODO: Replace the example implementation below with your own search
    #     #       method by combining techniques from lecture
    #     #
    #     # EXAMPLE: choose a random move without any search--this function MUST
    #     #          call self.queue.put(ACTION) at least once before time expires
    #     #          (the timer is automatically managed for you)
    #     import random
    #     self.queue.put(random.choice(state.actions()))

    def get_action(self, state):
        """ Choose an action available in the current state

        See RandomPlayer and GreedyPlayer for examples.

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        **********************************************************************
        NOTE: since the caller is responsible for cutting off search, calling
              get_action() from your own code will create an infinite loop!
              See (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # randomly select a move as player 1 or 2 on an empty board, otherwise
        # return the optimal minimax move at a fixed search depth of 3 plies
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            # TODO env var to select option
            
            # Iterative deepening - keep looping until we get terminated
            depth = 3
            while True:
                if os.environ.get('P3_PLAYER', '') == 'minimax':
                    self.queue.put(self.minimax(state, depth=depth))
                else:
                    self.queue.put(self.minimax_alpha_beta(state, depth=depth))
                if os.environ.get('P3_DEBUG', ''):
                    print('completed depth', depth)
                depth += 1

    def minimax(self, state, depth):

        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value
        
        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

    def minimax_alpha_beta(self, state, depth):

        def min_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        def max_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value
        
        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1,
                                                            float('-inf'), float('inf')))

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)