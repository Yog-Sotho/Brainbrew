import json
import numpy as np

def dataset_stats(path):

    lengths=[]

    with open(path) as f:
        for line in f:

            obj=json.loads(line)
            lengths.append(len(obj["output"].split()))

    return {
        "samples":len(lengths),
        "avg_tokens":float(np.mean(lengths)),
        "max_tokens":int(np.max(lengths)),
        "min_tokens":int(np.min(lengths))
    }
