

.PHONY: all clean install

all: install Non_adaptive_comparison toy_examples Deep_RL



Non_adaptive_comparison: Non_adaptive_comparison_xp/exp_true_positives.py
	( \
	source /tmp/adastop_venv/bin/activate ;\
	python3 Non_adaptive_comparison_xp/exp_true_positives.py $(ARG);\
	)

toy_examples: toy_examples_xp/simulateR_2.py toy_examples_xp/simulatedR_multi_1seed.py toy_examples_xp/plot_cases12.py
	( \
	source /tmp/adastop_venv/bin/activate ;\
	python3 toy_examples_xp/simulateR_2.py 1 $(ARG) ;\
	python3 toy_examples_xp/simulateR_2.py 2 $(ARG) ;\
	python3 toy_examples_xp/simulatedR_multi_1seed.py $(ARG) ;\
	python3 toy_examples_xp/simulatedR_multi.py $(ARG);\
	python3 toy_examples_xp/plot_cases12.py ;\
	)

Deep_RL: Deep_RL_xp/scripts/plot_mujoco_comparisons.py Deep_RL_xp/scripts/plot_mujoco_sample_efficiency.py
	( \
	source /tmp/adastop_venv/bin/activate ;\
	python3 Deep_RL_xp/scripts/plot_mujoco_comparisons.py --path Deep_RL_xp/experiments --draw-boxplot ;\
	python3 Deep_RL_xp/scripts/plot_mujoco_sample_efficiency.py --path Deep_RL_xp/experiments ;\
	)
install:
	( \
	python3 -m venv /tmp/adastop_venv ;\
	source /tmp/adastop_venv/bin/activate ;\
	pip3 install -r requirements.txt ;\
	)

clean:
	rm -f results/*
