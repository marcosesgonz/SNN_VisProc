#!/bin/bash

python LSTMRegressor.py

python LSTMAction.py

# stdbuf -oL ./script.sh  2>&1 | tee train_all_ts`date +%s`.log