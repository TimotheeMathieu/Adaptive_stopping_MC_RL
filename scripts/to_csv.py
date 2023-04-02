import os

import numpy as np
import pandas as pd

N, K = 5, 6

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    dirpath = os.path.dirname(args.path)
    env = os.path.basename(args.path)[:-4]
    results = np.load(args.path, allow_pickle=True).item()

    os.makedirs(os.path.join(dirpath, env), exist_ok=True)

    for k, v in results.items():
        print(k, len(v))

    for k in range(K):
        start = k * N

        df_k = dict()
        for algo, v in results.items():
            if len(v) <= start:
                continue
            end = min((k + 1) * N, len(v))
            df_k[algo] = pd.Series(v[start:end])

        if len(df_k) == 0:
            break

        df_k = pd.DataFrame(df_k)
        df_k.to_csv(os.path.join(dirpath, f'{env}/{env}{k+1}.csv'))
