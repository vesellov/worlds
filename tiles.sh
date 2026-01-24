#!/bin/bash

mkdir -p tiles
mkdir -p shapes
mkdir -p groups

if [ -z "$2" ]
then
  rm index.json
  rm -rf tiles/$1/*
  rm -rf shapes/$1/*
  rm -rf groups/$1/*
  rm -rf results/$1/*
  ./venv/bin/python3 src/tiles.py stage1 maps/$1 tiles/$1
  ./venv/bin/python3 src/tiles.py stage2 tiles/$1 groups/$1 0.92
  exit 1
fi

./venv/bin/python3 src/tiles.py stage3 tiles/$1 shapes/$1 0.4 12 $2
# read -p "check & fix files in shapes/$1 ..."

# ./venv/bin/python3 src/tiles.py stage3 tiles/$1 shapes/$1 0.6 8 $2
# read -p "check & fix files in shapes/$1 ..."

# ./venv/bin/python3 src/tiles.py stage3 tiles/$1 shapes/$1 0.5 8 $2
# read -p "check & fix files in shapes/$1 ..."

# ./venv/bin/python3 src/tiles.py stage3 tiles/$1 shapes/$1 0.4 8 $2
# read -p "check & fix files in shapes/$1 ..."

# ./venv/bin/python3 src/tiles.py stage4 shapes/$1/left_right groups/$1 0.4 left $2
# ./venv/bin/python3 src/tiles.py stage4 shapes/$1/left_right groups/$1 0.4 right $2
# ./venv/bin/python3 src/tiles.py stage4 shapes/$1/top_bottom groups/$1 0.4 top $2
# ./venv/bin/python3 src/tiles.py stage4 shapes/$1/top_bottom groups/$1 0.4 bottom $2

# ./venv/bin/python3 src/tiles.py stage5 shapes/$1/top_right groups/$1 0.4 top_right $2
# ./venv/bin/python3 src/tiles.py stage5 shapes/$1/top_left groups/$1 0.4 top_left $2
# ./venv/bin/python3 src/tiles.py stage5 shapes/$1/bottom_left groups/$1 0.4 bottom_left $2
# ./venv/bin/python3 src/tiles.py stage5 shapes/$1/bottom_right groups/$1 0.4 bottom_right $2
# read -p "check & fix files in groups/$1 ..."

# ./venv/bin/python3 src/tiles.py stage6 groups/$1 results/$1 0.5 left $2
# ./venv/bin/python3 src/tiles.py stage6 groups/$1 results/$1 0.5 right $2
# ./venv/bin/python3 src/tiles.py stage6 groups/$1 results/$1 0.5 top $2
# ./venv/bin/python3 src/tiles.py stage6 groups/$1 results/$1 0.5 bottom $2

# ./venv/bin/python3 src/tiles.py stage7 groups/$1 results/$1 0.5 top_left $2
# ./venv/bin/python3 src/tiles.py stage7 groups/$1 results/$1 0.5 top_right $2
# ./venv/bin/python3 src/tiles.py stage7 groups/$1 results/$1 0.5 bottom_left $2
# ./venv/bin/python3 src/tiles.py stage7 groups/$1 results/$1 0.5 bottom_right $2
# read -p "check & fix files in results/$1 ..."

./venv/bin/python3 src/tiles.py stage8 shapes/$1 groups/$1 merged/$1 ready
