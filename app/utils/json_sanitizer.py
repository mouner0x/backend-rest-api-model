import math
import numpy as np

def sanitize_for_json(obj):
    """
    Recursively sanitize any object to be JSON serializable.
    - Convert numpy types to Python native types
    - Replace NaN, inf, -inf with 0.0
    """

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]

    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(i) for i in obj)

    elif isinstance(obj, (np.integer,)):
        return int(obj)

    elif isinstance(obj, (np.floating,)):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)

    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj

    elif obj is None:
        return None

    else:
        return obj
