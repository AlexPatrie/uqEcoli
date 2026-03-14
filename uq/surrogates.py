import pickle
from pathlib import Path


def export_instance(instance, path: Path):
    path.write_bytes(pickle.dumps(instance))
