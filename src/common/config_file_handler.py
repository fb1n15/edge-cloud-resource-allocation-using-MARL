import yaml
from ray import tune


def load_yaml(filename, tune_wrap=True):
    with open(filename) as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)
        if tune_wrap:
            yml = do_tune_wrap(yml)
        return yml


def do_tune_wrap(d):
    if hasattr(d, "items"):
        # If it is dict like
        r = {}
        for key, value in d.items():
            if key == "loguniform":
                return tune.loguniform(*value)
            elif key == "uniform":
                return tune.uniform(*value)
            elif key == "choice":
                return tune.choice([*value])
            else:
                r[key] = do_tune_wrap(value)
        return r
    else:
        return d
