#!/bin/bash
echo "Seed: $1"
source activate as_crl
python ~/Adaptive_stopping_MC_RL/adastop/deep_rl_agents/ddpg.py -e Ant-v3 -b 2000000 -s $1
