#!/bin/bash
echo "Seed: $1"
source activate as_rlb
python ~/Adaptive_stopping_MC_RL/adastop/deep_rl_agents/ppo.py -e Walker2d-v3 -s $1