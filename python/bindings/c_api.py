import ctypes
import os
from pathlib import Path

import numpy as np

LINEAR_TASK_REGRESSION = 0
LINEAR_TASK_CLASSIFICATION = 1

MLP_TASK_CLASSIFICATION = 1


def load_library() -> ctypes.CDLL:
    if os.name == "nt":
        mingw_bin = Path(r"C:\mingw64\bin")
        if mingw_bin.exists():
            os.add_dll_directory(str(mingw_bin))

    # cmake_build = "cmake-build-debug"
    cmake_build = "cmake-build-release"

    project_root = Path(__file__).resolve().parents[2]
    dll_path = project_root / cmake_build / "libpa_ml.dll"

    if not dll_path.exists():
        raise FileNotFoundError(f"Bibliothèque introuvable: {dll_path}")

    return ctypes.CDLL(str(dll_path))


def configure_api(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.my_add.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.my_add.restype = ctypes.c_int32

    lib.create_linear_model.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.create_linear_model.restype = ctypes.c_void_p

    lib.train_linear_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_int32,
    ]
    lib.train_linear_model.restype = ctypes.c_int32

    lib.predict_linear_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.predict_linear_model.restype = ctypes.c_int32

    lib.destroy_linear_model.argtypes = [ctypes.c_void_p]
    lib.destroy_linear_model.restype = None

    lib.create_mlp_model.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
    ]
    lib.create_mlp_model.restype = ctypes.c_void_p

    lib.train_mlp_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_int32,
    ]
    lib.train_mlp_model.restype = ctypes.c_int32

    lib.predict_mlp_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.predict_mlp_model.restype = ctypes.c_int32

    lib.destroy_mlp_model.argtypes = [ctypes.c_void_p]
    lib.destroy_mlp_model.restype = None

    lib.sum_array.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    lib.sum_array.restype = ctypes.c_float

    return lib


def as_double_pointer(values: np.ndarray):
    array = np.ascontiguousarray(values, dtype=np.float64)
    pointer = array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    return array, pointer


def as_int32_pointer(values: np.ndarray):
    array = np.ascontiguousarray(values, dtype=np.int32)
    pointer = array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    return array, pointer


lib = configure_api(load_library())


def run() -> None:
    print(lib.my_add(33, 22))

    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([[1.0], [3.0], [5.0], [7.0]], dtype=np.float64)
    x_test = np.array([4.0], dtype=np.float64)
    y_pred = np.zeros(1, dtype=np.float64)

    x_array, x_ptr = as_double_pointer(x)
    _, y_ptr = as_double_pointer(y)
    _, x_test_ptr = as_double_pointer(x_test)
    y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

    model = lib.create_linear_model(1, 1, LINEAR_TASK_REGRESSION)
    if not model:
        raise RuntimeError("Impossible de créer le modèle linéaire.")

    try:
        train_status = lib.train_linear_model(
            model,
            x_ptr,
            y_ptr,
            x_array.shape[0],
            0.01,
            1000,
        )
        if train_status != 0:
            raise RuntimeError(f"Erreur pendant l'entraînement: {train_status}")

        predict_status = lib.predict_linear_model(model, x_test_ptr, y_pred_ptr)
        if predict_status != 0:
            raise RuntimeError(f"Erreur pendant la prédiction: {predict_status}")

        print(float(y_pred_array[0]))

    finally:
        lib.destroy_linear_model(ctypes.c_void_p(model))

    values = np.array([66.0, 44.0], dtype=np.float32)
    values_pointer = np.ctypeslib.as_ctypes(values)
    print(lib.sum_array(values_pointer, len(values)))


if __name__ == "__main__":
    run()