#!/bin/bash
# this script will create data for signed digits data for use with mobilenet.
# after running try.sh, you should run 'python3 tf_mobilenet.ph'

set -x
set -o nounset


IN_DIR="$HOME/Downloads/data/sd"

cd $IN_DIR
rm -rf $IN_DIR/*
EXEC_DIR="$HOME/g/python3"


git clone https://github.com/ardamavi/Sign-Language-Digits-Dataset.git
cd $EXEC_DIR
OUT_DIR="$EXEC_DIR/data/sd"
mkdir --parents $OUT_DIR

echo "making OUTPUT dir for mobilenet data: " `pwd`
mkdir --parents $OUT_DIR/train/
chmod 777 $OUT_DIR/train

#these mkdir commands don't seem to work from a shell.
#i had to execute them by hand at the shell prompt

mkdir --parents $OUT_DIR/train/0/
mkdir --parents $OUT_DIR/train/1/
mkdir --parents $OUT_DIR/train/2/
mkdir --parents $OUT_DIR/train/3/
mkdir --parents $OUT_DIR/train/4/
mkdir --parents $OUT_DIR/train/5/
mkdir --parents $OUT_DIR/train/6/
mkdir --parents $OUT_DIR/train/7/
mkdir --parents $OUT_DIR/train/8/
mkdir --parents $OUT_DIR/train/9/
mkdir --parents $OUT_DIR/test/
chmod 777 $OUT_DIR/test

mkdir --parents $OUT_DIR/test/0/
mkdir --parents $OUT_DIR/test/1/
mkdir --parents $OUT_DIR/test/2/
mkdir --parents $OUT_DIR/test/3/
mkdir --parents $OUT_DIR/test/4/
mkdir --parents $OUT_DIR/test/5/
mkdir --parents $OUT_DIR/test/6/
mkdir --parents $OUT_DIR/test/7/
mkdir --parents $OUT_DIR/test/8/
mkdir --parents $OUT_DIR/test/9/
mkdir --parents $OUT_DIR/valid
chmod 777 $OUT_DIR/valid

mkdir --parents $OUT_DIR/valid/0/
mkdir --parents $OUT_DIR/valid/1/
mkdir --parents $OUT_DIR/valid/2/
mkdir --parents $OUT_DIR/valid/3/
mkdir --parents $OUT_DIR/valid/4/
mkdir --parents $OUT_DIR/valid/5/
mkdir --parents $OUT_DIR/valid/6/
mkdir --parents $OUT_DIR/valid/7/
mkdir --parents $OUT_DIR/valid/8/
mkdir --parents $OUT_DIR/valid/9/
# run the python program
python3 ./data_mobilenet.py




