#!/bin/bash
echo "Seed: $1"
source activate as_rlb
python ~/Adaptive_stopping_MC_RL/adastop/deep_rl_agents/ppo.py -e Ant-v3 -b 2000000 -s $1