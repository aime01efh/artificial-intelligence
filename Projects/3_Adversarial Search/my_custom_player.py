import random
import os
import math
from sample_players import DataPlayer
from isolation import DebugState


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

        Environment variables:
            - P3_PLAYER: "minimax": minimax w/iter deep but no alpha-beta pruning
                         "minimax_ab": minimax w/iter deep and alpha-beta pruning
                         empty: Monte Carlo Tree Search
            - P3_DEBUG: non-empty to enable debugging output
        """
        if os.environ.get('P3_PLAYER', '') in ['minimax', 'minimax_ab']:
            self.iterative_deepening(state)
        else:
            self.monte_carlo_tree_search(state)

    def iterative_deepening(self, state):
        # randomly select a move as player 1 or 2 on an empty board, otherwise
        # return the optimal minimax move at a fixed search depth of 3 plies
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            # Iterative deepening - keep looping until we get terminated
            depth = 3
            while True:
                p3_player = os.environ.get('P3_PLAYER', '')
                if p3_player == 'minimax':
                    self.queue.put(self.minimax(state, depth=depth))
                elif p3_player == 'minimax_ab':
                    self.queue.put(self.minimax_alpha_beta(state, depth=depth))
                else:
                    raise ValueError(f'Unknown P3_PLAYER {p3_player}')
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
        
        if os.environ.get('P3_DEBUG', ''):
            print(f'Starting minimax for player {state.player()}')
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
        
        if os.environ.get('P3_DEBUG', ''):
            print(f'Starting minimax_alpha_beta for player {state.player()}')
        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1,
                                                            float('-inf'), float('inf')))

    def monte_carlo_tree_search(self, state, update_every=50):
        """Run a Monte Carlo Tree Search with root at the given state, providing
        the current best action every "update_every" iterations
        """
        if os.environ.get('P3_DEBUG', ''):
            print(f'Starting MCTS for player {state.player()}')
        tree = CustomPlayer.MCTSNode(state, state.player(), parent=None, relative_ply=0)
        if len(tree.actions) == 1:
            if os.environ.get('P3_DEBUG', ''):
                print('  only one action to choose')
            return tree.action[0]

        loopcount = 0
        while True:
            #if os.environ.get('P3_DEBUG', ''):
                #print(f'  MCTS loopcount {loopcount}')
            leaf = tree.select()
            child = leaf.expand()
            result = child.simulate()
            child.backpropagate(result)
            loopcount += 1
            if loopcount >= update_every:
                best_action = tree.best_action()
                self.queue.put(best_action)
                if os.environ.get('P3_DEBUG', ''):
                    print(f'  MCTS provided action {best_action} after {tree.num_playouts} playouts: '
                          f'{[x.num_wins for x in tree.action_nodes[:20]]}')
                    # print(tree.actions)
                    try:
                        state.result(best_action)
                    except:
                        print('oops')
                loopcount = 0

    class MCTSNode:
        """Node class for Monte Carlo Tree Search graph using UCT
        """
        def __init__(self, state, player_id, parent, relative_ply):
            self.state = state
            self.player_id = player_id
            self.relative_ply = relative_ply
            self.parent = parent
            self.actions = []
            self.action_nodes = []
            self.action_node_ucb1s = []

            self.num_wins = self.num_playouts = 0
            self.ucb1_c = math.sqrt(2.0)

        def __str__(self):
            return (f'  MCTSNode(player={self.player_id}, relply={self.relative_ply}, '
                   f'numchildren={len(self.actions)}, wins={self.num_wins}, plays={self.num_playouts}')

        def best_action(self):
            """Return the action with the most playouts"""
            child_playouts = [x.num_playouts for x in self.action_nodes]
            selected_idx = child_playouts.index(max(child_playouts))
            return self.actions[selected_idx]

        def select(self):
            """Recursive UCB1-based selection to locate a leaf node.  expand() creates all child
            nodes so we look for the absence of any children to detect a leaf.
            """
            if not self.actions:
                return self
            #if os.environ.get('P3_DEBUG', ''):
                #print(f'  select(): {str(self)}, {str(self.action_node_ucb1s)}')
                #print(f'  select(): {str(self)}')
            selected_idx = self.action_node_ucb1s.index(max(self.action_node_ucb1s))
            selected_child = self.action_nodes[selected_idx]
            return selected_child.select()
        
        def expand(self):
            """Create and initialize all child nodes of the current leaf node.
            Return the first child node.
            """
            self.actions = self.state.actions()
            #if os.environ.get('P3_DEBUG', ''):
                #print(f'  Expanding {str(self)} to {len(self.actions)} actions')
                #print(DebugState.from_state(self.state))
            for action in self.actions:
                child_state = self.state.result(action)
                self.action_nodes.append(CustomPlayer.MCTSNode(child_state, self.player_id,
                                                               self, self.relative_ply + 1))
                self.action_node_ucb1s.append(float('inf'))
            
            if not self.action_nodes:
                # terminal node, no children
                return self

            return self.action_nodes[0]
        
        def simulate(self):
            """Run a playout on the current node using a random playout policy"""
            #if os.environ.get('P3_DEBUG', ''):
                #print(f'  simulate(): {str(self)}')
            state = self.state
            while not state.terminal_test():
                random_action = random.choice(state.actions())
                state = state.result(random_action)
            return state.utility(self.player_id)
        
        def backpropagate(self, result):
            """Update leaf and all ancestors with the results of the playout"""
            node = self
            # if os.environ.get('P3_DEBUG', ''):
            #     print('  backpropagate()')
            while node:
                node.num_playouts += 1
                
                if (node.state.player() == node.player_id) != (result > 0):
                    # Inc node wins when this node player is same as original and result
                    # is a loss, or if node player is different and it's a win
                    node.num_wins += 1
                    # if os.environ.get('P3_DEBUG', ''):
                    #     print(f'    backprop win: {str(node)}, stateplayer={node.state.player()}, result={result}')
                for idx, child_node in enumerate(node.action_nodes):
                    node.action_node_ucb1s[idx] = node.compute_ucb1(child_node)
                # if os.environ.get('P3_DEBUG', ''):
                #     print(f'    updated       {str(node)}: {str(node.action_node_ucb1s)}')
                node = node.parent
            # if os.environ.get('P3_DEBUG', ''):
            #     print('---')

        def compute_ucb1(self, child_node):
            """Compute the UCB1 score of the given child node"""
            if child_node.num_playouts == 0:
                return float('inf')
            
            exploitation = child_node.num_wins / child_node.num_playouts
            exploration = math.sqrt(math.log(self.num_playouts) / child_node.num_playouts)
            ucb1 = exploitation + self.ucb1_c * exploration
            return ucb1
            
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
