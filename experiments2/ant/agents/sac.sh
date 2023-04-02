#!/bin/bash
echo "Seed: $1"
source activate as_sb
python python ~/Adaptive_stopping_MC_RL/adastop/deep_rl_agents/sac.py -e Ant-v3 -b 1 -s $1