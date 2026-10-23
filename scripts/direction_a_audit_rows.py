"""JSON pair normalization at the audit boundary; no scoring rule changes."""
import hashlib
from functools import wraps

def normalizing_metric_row(function):
    @wraps(function)
    def normalized(sid,record,pred,base=None):
        return function(sid,record,set(map(tuple,pred)),set(map(tuple,base)) if base is not None else None)
    return normalized

def verify_dependencies(root,manifest):
    for name,expected in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,'Audit executable source changed: '+name

