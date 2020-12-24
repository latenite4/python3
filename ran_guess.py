#!/usr/bin/python3

import random

#pyton3 examples from https://www.youtube.com/watch?v=8ext9G7xspg&list=PLZ6uU-qwmL8Sv78TscECM6kqb-zIxlgdY&index=10&t=240s
#name: R. Melton
#date: 12/20/20


#python routine to guess the number which the computer has chosen.
def guess(x):
  random_number = random.randint(1,x)
  guess = 0
  while guess != random_number:
    guess = int(input(f'Guess number between 1 and {x}: '))
    if guess < random_number:
      print('Sorry, too low')
    elif guess > random_number:
      print('Sorry, too high')
  
  print(f'yay, you got it { random_number}')

def computer_guess (x):
  low = 1
  high = x
  feedback = ''
  guess = random.randint(low,high)

  print(f'secret number is {guess}')

  while feedback != 'c':
    if low != high:
      guess = random.randint(low,high)
    else:
      guess = low

    guess = random.randint(low,high)
    feedback = input (f'is {guess } too high (H), too low (L), or correct (C)?').lower()
    if feedback == 'h':
      high = guess - 1
    elif feedback == 'l':
      low = guess + 1
  
  print(f'yay, the computer guessed correctly {guess}!')


computer_guess(10)