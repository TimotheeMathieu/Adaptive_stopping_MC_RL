#!/bin/bash
echo "Seed: $1"
source activate as_mrl
python ~/Adaptive_stopping_MC_RL/adastop/deep_rl_agents/trpo.py -e HalfCheetah-v3 -s $1