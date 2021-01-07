#!/bin/bash

set -x
cd $HOME/Downloads/data/signed_digits
rm -rf *
git clone https://github.com/ardamavi/Sign-Language-Digits-Dataset.git
cd $HOME/g/python3
rm -rf data/sd
cd $HOME/g/python3
./clean.sh
python3 ./data_mobilenet.py




