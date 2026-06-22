import ctypes
import os
from pathlib import Path

import numpy as np


def load_library():
    os.add_dll_directory(r"C:\mingw64\bin")

    project_root = Path(__file__).resolve().parents[2]
    dll_path = project_root / "cmake-build-debug" / "libpa_ml.dll"

    if not dll_path.exists():
        raise FileNotFoundError(f"Bibliothèque introuvable: {dll_path}")

    return ctypes.CDLL(str(dll_path))


def run():
    lib = load_library()

    lib.my_add.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.my_add.restype = ctypes.c_int32

    lib.create_linear_model.argtypes = [ctypes.c_float, ctypes.c_float]
    lib.create_linear_model.restype = ctypes.c_void_p

    lib.predict_linear_model.argtypes = [ctypes.c_void_p]
    lib.predict_linear_model.restype = ctypes.c_float

    lib.release_linear_model.argtypes = [ctypes.c_void_p]
    lib.release_linear_model.restype = None

    lib.sum_array.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    lib.sum_array.restype = ctypes.c_float

    print(lib.my_add(33, 22))

    model = lib.create_linear_model(42.0, 51.0)
    print(lib.predict_linear_model(model))
    lib.release_linear_model(model)

    values = np.array([66.0, 44.0], dtype=np.float32)
    values_pointer = np.ctypeslib.as_ctypes(values)
    print(lib.sum_array(values_pointer, len(values)))


if __name__ == "__main__":
    run()