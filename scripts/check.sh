#!/bin/sh

make evaluate
python3 ./scripts/check.py --reference  ./results/reference/evaluate.dat --result ./results/final_state.dat

make visual
python3 ./scripts/check.py --reference  ./results/reference/visual.dat --result ./results/final_state.dat