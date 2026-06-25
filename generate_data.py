import random
import json
import multiprocessing


class UTTTGame:
    """Simplified Game Engine for Data Generation."""

    def __init__(self):
        self.boards = [[None] * 9 for _ in range(9)]
        self.global_board = [None] * 9
        self.turn = 'X'
        self.next_board = None

    def get_winner(self, board):
        """Checks for a winner in a 3x3 grid."""
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for line in lines:
            if board[line[0]] == board[line[1]] == board[line[2]] != None:
                return board[line[0]]
        return None

    def is_board_locked(self, b_idx):
        """Checks if a local board is won or full."""
        return (self.get_winner(self.boards[b_idx]) is not None or
                None not in self.boards[b_idx])

    def get_legal_moves(self):
        """Returns all legal moves based on UTTT rules."""
        playable = [i for i in range(9) if not self.is_board_locked(i)]
        target = [self.next_board] if (self.next_board is not None and
                                      self.next_board in playable) else playable
        moves = []
        for b_idx in target:
            for c_idx in range(9):
                if self.boards[b_idx][c_idx] is None:
                    moves.append((b_idx, c_idx))
        return moves

    def make_move(self, move):
        """Executes a move and updates state."""
        b_idx, c_idx = move
        self.boards[b_idx][c_idx] = self.turn
        local_win = self.get_winner(self.boards[b_idx])
        if local_win:
            self.global_board[b_idx] = local_win
        self.next_board, self.turn = c_idx, ('O' if self.turn == 'X' else 'X')

    def get_state(self):
        """Returns the current state as a hashable tuple."""
        return (tuple(tuple(b) for b in self.boards),
                tuple(self.global_board),
                self.turn,
                self.next_board)


class TeacherBot:
    """A bot that uses limited depth search to ensure high-quality data."""

    def choose_move(self, game):
        """Chooses a move prioritizing local wins over random play."""
        moves = game.get_legal_moves()
        if not moves:
            return None

        for move in moves:
            if self.is_winning_move(game, move):
                return move

        return random.choice(moves)

    def is_winning_move(self, game, move):
        """Checks if a specific move wins the current local board."""
        b_idx, c_idx = move
        temp_board = list(game.boards[b_idx])
        temp_board[c_idx] = game.turn
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for line in lines:
            if (temp_board[line[0]] == temp_board[line[1]] ==
                    temp_board[line[2]] == game.turn):
                return True
        return False


def worker_task(num_games, filename, lock):
    """
    Simulates a subset of games and writes them to disk using a shared lock.
    Each process manages its own TeacherBot instance.
    """
    bot = TeacherBot()
    local_buffer = []
    buffer_limit = 100  # Store 100 game-histories before writing to reduce I/O

    for g_idx in range(num_games):
        game = UTTTGame()
        history = []

        while True:
            state = game.get_state()
            move = bot.choose_move(game)
            if move is None or game.get_winner(game.global_board) is not None:
                break
            history.append(state)
            game.make_move(move)

        winner = game.get_winner(game.global_board)
        outcome = 1.0 if winner == 'X' else (-1.0 if winner == 'O' else 0.0)

        # Add the history of this game to the local buffer
        for state in history:
            local_buffer.append(json.dumps({"state": state, "outcome": outcome}))

        # If buffer is full, acquire lock and dump to disk
        if len(local_buffer) >= (buffer_limit * 50):  # Rough estimate of moves/game
            with lock:
                with open(filename, "a") as f:
                    f.write("\n".join(local_buffer) + "\n")
                local_buffer.clear()

    # Final dump for remaining items in buffer
    if local_buffer:
        with lock:
            with open(filename, "a") as f:
                f.write("\n".join(local_buffer) + "\n")


def generate_training_set_parallel(num_games=10000000,
filename="uttt_train_data.jsonl"):
    """
    Sets up multiprocessing to simulate games across all available CPU cores.
    """
    # Initialize file (clear existing)
    with open(filename, "w") as f:
        f.write("")

    cpu_count = multiprocessing.cpu_count()
    games_per_worker = num_games // cpu_count
    remainder = num_games % cpu_count

    # Use a Manager to create a shared lock across processes
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    print(f"Spawning {cpu_count} workers for {num_games} games...")
    processes = []
    for i in range(cpu_count):
        # Distribute the remainder to the first worker
        count = games_per_worker + (remainder if i == 0 else 0)
        p = multiprocessing.Process(
            target=worker_task,
            args=(count, filename, lock)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"Completed. Data saved to {filename}")


if __name__ == "__main__":
    # Set the number of games you wish to generate
    generate_training_set_parallel(num_games=10000000)
