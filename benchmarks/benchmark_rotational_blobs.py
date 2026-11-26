"""Paired CPU ablation on Gaussian blobs. Run from repository root as a module."""
import argparse
import csv
from itertools import product

import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from smeans._cpu import fit_cpu, tiled_assign
from smeans._recipe import SMeansConfig

VARIANTS = {
    'baseline': SMeansConfig(),
    'rotational_5': SMeansConfig(extrapolation=True),
    'rotational_1': SMeansConfig(extrapolation=True, extrapolation_interval=1),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=int, default=20000)
    p.add_argument('--test', type=int, default=5000)
    p.add_argument('--clusters', type=int, default=32)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output', default='rotational_blobs.csv')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    fields = ['d', 'std', 'epochs', 'seed', 'variant', 'test_mse', 'train_mse', 'time_ms', 'steps', 'splits']
    with open(a.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d, std in product((8, 64), (1.0, 4.0)):
            x, y = make_blobs(n_samples=a.train+a.test, centers=a.clusters,
                              n_features=d, cluster_std=std, random_state=42)
            train, test = train_test_split(x, test_size=a.test, stratify=y, random_state=43)
            train, test = (torch.as_tensor(z, dtype=torch.float32) for z in (train, test))
            fit_cpu(train[:max(512, a.clusters)], a.clusters, nepochs=0.25)
            for epochs, seed in product((0.25, 0.5, 1.0, 2.0), range(a.seeds)):
                names = list(VARIANTS)
                names = names[seed % len(names):] + names[:seed % len(names)]
                for name in names:
                    r = fit_cpu(train, a.clusters, nepochs=epochs, seed=seed, cfg=VARIANTS[name])
                    _, dist = tiled_assign(test, r['centroids'])
                    w.writerow(dict(d=d, std=std, epochs=epochs, seed=seed, variant=name,
                                    test_mse=float(dist.mean()), train_mse=r['inertia']/len(train),
                                    time_ms=1000*r['time_s'], steps=r['n_iter'], splits=r['total_splits']))
                f.flush()


if __name__ == '__main__':
    main()
