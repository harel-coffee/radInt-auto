#!/bin/bash

# clear
rm -rf results
rm -rf paper
mkdir results
mkdir paper

# start experiments
python3 ./startExperiment.py

# evaluate
python3 ./evaluate.py


#
