# group: 2014480, 2035469, 851623
# Yong Kien Lin, Sandesh Adhikari, Janhavi Singhal

# Code Experimentation/Thoughts:

# We considered picking the Monte Carlo algorithm, however, we had to take into consideration its time and space complexity 
# that increases exponentially with the increase of search depth. It would be ideal with a more proper depth choice and 
# random sampling strategy, otherwise, it wastes time. Monte Carlo also needs a large number of random simulations to get 
# the optimal performance, which is time-consuming.

# Another algorithm that could prove to be the most effective based on some studies, is the Threat-space search which 
# utilises just focusing on the attacking moves and ignores any defensive plays that are made, however, there was not 
# enough information and time for us to be able to implement this.

# Therefore, we picked the minimax algorithm (with alpha-beta pruning) because the base minimax algorithm goes through all 
# the possibilities which makes the search space too large to explore; however, with alpha-beta pruning, it decreases the number 
# of nodes that are evaluated through skipping unnecessary branches. Therefore making the algorithm more efficient and less time- 
# consuming.

import numpy as np

from misc import legalMove
from gomokuAgent import GomokuAgent

# Players for minimax.
PLAYER_BLACK = 1
PLAYER_WHITE = -1

# Overridden move function.
class Player(GomokuAgent):
    
    def move(self, board):
        while True:
            moveLoc, _ = minimax(board, self.ID, 1, 0, 0, self.X_IN_A_LINE)
            
            if(moveLoc == None):
                moveLoc = tuple(np.random.randint(self.BOARD_SIZE, size=2))

            if legalMove(board, moveLoc):
                return moveLoc

# Function which calculates the score of a given board state, based on the number of open lines + 1.

# For example for the horizontal line (Where -1 is the opponent):
# 0 0 0 0 0 0 0 0 0 0 0 = Open Line, score incremented.
# 0 0 0 0 0 -1 0 0 0 0 0 = Non-open line, score not incremented.

# This thinking then is the same for all cardinal directions.

def gomoku_evaluation(board, p, WIN_LENGTH):
    BOARD_SIZE = board.shape[0]
# Count the number of open lines.
    score = 0
    #Loop through the board.
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0:
                # Check horizontal line.
                if j <= BOARD_SIZE - WIN_LENGTH and \
                   np.sum(board[i][j:j+WIN_LENGTH]) == p * WIN_LENGTH:
                    score += 1
                # Check vertical line.
                if i <= BOARD_SIZE - WIN_LENGTH and \
                   np.sum(board[i:i+WIN_LENGTH,j]) == p * WIN_LENGTH:
                    score += 1
                # Check diagonal line.
                if i <= BOARD_SIZE - WIN_LENGTH and j <= BOARD_SIZE - WIN_LENGTH and \
                   np.sum([board[i+k][j+k] for k in range(WIN_LENGTH)]) == p * WIN_LENGTH:
                    score += 1
                # Check anti-diagonal line.
                if i >= WIN_LENGTH - 1 and j <= BOARD_SIZE - WIN_LENGTH and \
                   np.sum([board[i-k][j+k] for k in range(WIN_LENGTH)]) == p * WIN_LENGTH:
                    score += 1

    return score

# Minimax algorithm with alpha-beta pruning.
def minimax(board, p, depth, alpha, beta, WIN_LENGTH):
    BOARD_SIZE = board.shape[0]
    # Check if game is over or depth limit reached.
    if np.abs(gomoku_evaluation(board, PLAYER_BLACK, WIN_LENGTH)) >= 1 or np.abs(gomoku_evaluation(board, PLAYER_WHITE, WIN_LENGTH)) >= 1 or depth == 0:
        return None, gomoku_evaluation(board, p, WIN_LENGTH)
    
    best_move = None
    best_score = -np.inf if p == PLAYER_BLACK else np.inf
    
    # Loop over all possible moves.
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0:
                # Make the move
                board[i][j] = p
                # Recursive call to minimax, so we can traverse the tree.
                _, score = minimax(board, -p, depth -1, alpha, beta, WIN_LENGTH)
                # Undo the move.
                board[i][j] = 0
                # Update best score and move.
                # For max.
                if p == PLAYER_BLACK:
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
                    alpha = max(alpha, score)
                else:
                # For min.
                    if score < best_score:
                        best_score = score
                        best_move = (i, j)
                    beta = min(beta, score)
                # Alpha-beta pruning.
                if alpha >= beta:
                    break
        if alpha >= beta:
            break

    return best_move, best_score


