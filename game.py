
from player import HumanPlayer, RandomComputerPlayer
import time, math

class TicTacToe:
  def __init__(self):
    self.board = self.make_board()
    self.current_winner = None

  def print_board(self):
    for row in [self.board[i*3:(i+1)*3] for i in range (3)]:
      print('| '+' | '.join(row) + ' |')

  @staticmethod
  def print_board_nums():
    number_board = [[str(i) for i in range(j*3,(j+1)*3)] for j in range(3)]
    for row in number_board:
      print('| '+' | '.join(row)+' |')

  def available_moves(self):
    return [i for i, x in enumerate(self.board) if x == " "]
  
  @staticmethod
  def make_board():
    return [' ' for _ in range(9)]

  def empty_squares(self):
    #rerurn bool
    return ' ' in self.board
  
  def num_empty_squares(self):
    return self.board.count(' ')

  def make_move(self,  square, letter):
    #if valid move, make move
    if self.board[square] == ' ':
      self.board[square] = letter
      if self.winner(square, letter):
        self.current_winner = letter
      return True
    return False

  def other_player(self,letter):
    if letter == 'O':
      return 'X'
    if letter == 'X':
      return 'O'
    print ("no letter found "+letter)
    return 'O'

  def winner(self, square, letter):
    #3 in a row anywhere?
    row_ind = math.floor(square / 3)
    row = self.board[row_ind*3 : (row_ind+1)*3]
    if all([s == letter for s in row]):
      return True
    col_ind = square % 3
    column = [self.board[col_ind+i*3]for i in range(3)]
    if all([s == letter for s in column]):
      return True
    #check diags
    if square % 2 == 0:
      diagonal1 = [self.board[i] for i in [0,4,8]]
      if all ([sorted == letter for s in diagonal1]):
        return True

      diagonal2 = [self.board[i] for i in [2,4,6]]
      if all ([s == letter for s in diagonal2]):
        return True
      #if all prev checks fail, no winner
      return False



def play(game, x_player,o_player,print_game=True):
  if print_game:
    game.print_board_nums()

  letter = 'X'
  #iterate while game has empty squares
  while game.empty_squares():
    if letter == 'O':
      square = o_player.get_move(game)
    else:
      square = x_player.get_move(game)

    if game.make_move(square,letter):
      if print_game:
        print(letter +f' makes move to {square}')
        game.print_board()
        print ('') #new line
      if game.current_winner:
        if print_game:
          print('winner is '+game.current_winner)
        return letter

      letter = 'O' if letter == 'X' else 'X' # switch player
    
    time.sleep(.8)

    if print_game and not game.empty_squares():
      print('it\'s a tie!')

if __name__ == '__main__':
  x_player = HumanPlayer('X')
  o_player = RandomComputerPlayer('O')
  t = TicTacToe()
  #TicTacToe().print_board_nums()
  play(t,x_player,o_player,print_game=True)