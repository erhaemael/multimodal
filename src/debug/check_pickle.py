import os
import pickle

def explore_structure(obj, indent=0, max_depth=10):
    """
    Rekursif untuk menjelajahi struktur objek Python (dict, list, tuple, np.array, dsb.)
    """
    prefix = "  " * indent
    if indent > max_depth:
        print(f"{prefix}... (max depth reached)")
        return

    if isinstance(obj, dict):
        print(f"{prefix}dict with {len(obj)} keys:")
        for k, v in obj.items():
            print(f"{prefix}  [{repr(k)}] -> {type(v).__name__}")
            explore_structure(v, indent + 2, max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} with {len(obj)} elements")
        if len(obj) > 0:
            print(f"{prefix}  [0] -> {type(obj[0]).__name__}")
            explore_structure(obj[0], indent + 2, max_depth)
    elif hasattr(obj, 'shape'):
        print(f"{prefix}{type(obj).__name__} with shape {obj.shape}, dtype={getattr(obj, 'dtype', 'unknown')}")
    else:
        print(f"{prefix}{type(obj).__name__}: {repr(obj)[:60]}")

def inspect_pickle_file(pickle_path):
    print(f"\nInspecting: {pickle_path}")
    if not os.path.exists(pickle_path):
        print("  File not found.")
        return

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f, encoding="latin1")

    explore_structure(data)

# Contoh penggunaan
if __name__ == "__main__":
    # Ganti dengan ID subjek / nama file sesuai kebutuhan
    patient_id = "S10"
    dataset_path = "data/raw/WESAD"
    pickle_file = os.path.join(dataset_path, patient_id, f"{patient_id}.pkl")

    inspect_pickle_file(pickle_file)
