#!/bin/bash
echo "Seed: $1"
python -c "import numpy as np;rng = np.random.default_rng($1);print(f'AdaStop Evaluation: {0.25 + rng.uniform()}')"
