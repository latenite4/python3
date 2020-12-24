#!/usr/bin/python3
#python program to play hangmen
#name: R. Melton
#date: 12/20/20

import random
from words import words
import string

def get_valid_word(words):
  word = random.choice(words)
  while '-' in word or ' ' in word:
    word = random.choice(words)
  return word.upper()

def hangman():
  lives =  6

  word = get_valid_word(words)
  word_letters = set(word)
  alphabet = set(string.ascii_uppercase)
  used_letters = set()    # what user has guessed

  while len(word_letters)>0 and lives > 0:
    print('You have', lives, 'lives left and you have used these letters: ', ' '.join(used_letters))
    word_list = [letter if letter in used_letters else '-' for letter in word]
    print('current word ',' '.join(word_list))
    print(f'you have {lives} lives left')

    user_letter = input(f'guess a letter: ').upper()

    if user_letter in alphabet - used_letters:
      used_letters.add(user_letter)
      if user_letter in word_letters:
        word_letters.remove(user_letter)
      else:
        lives = lives - 1
        print(f'letter {user_letter} not in the word')


    elif user_letter in used_letters:
      print('already used ')
    else:
      print('invalid, guess again')

    if lives == 0:
      print(f'sorry, you died, word was {word}')
    else:
      print(f'you guessed it {word}!')

if __name__ == '__main__':
  hangman()
