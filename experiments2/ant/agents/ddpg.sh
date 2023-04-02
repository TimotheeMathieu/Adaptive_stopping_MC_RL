#!/bin/bash
echo "Seed: $1"
source activate as_crl
python ~/adastop/Adaptive_stopping_MC_RL/adastop/deep_rl_agents/ddpg.py -e Ant-v3 -b 1 -s $1
