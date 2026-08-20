import random

def split_pairs(pairs, train_ratio = 0.7, val_ratio = 0.1, test_ratio = 0.2, seed = 42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "ratios must sum to 1.0"

    pairs = pairs[:]
    random.Random(seed).shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]

    print(f"Total: {n} | train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return {"train": train, "val": val, "test": test}