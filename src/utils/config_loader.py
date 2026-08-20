import os
import yaml

def get_config(path):
    config = None
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print("Loaded config", path)
    return config
        