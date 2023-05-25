#!/bin/sh

# Author : Ameya
obj=gzsl
fld=0
python3 train.py --objective $obj --fold $fld --eval

for obj in gzsl zsl
do
  for fld in 1 2 3
    do
      python3 train.py --objective $obj --fold $fld --eval
    done
done