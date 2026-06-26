import ctypes
import os
from pathlib import Path

import numpy as np

LINEAR_TASK_REGRESSION = 0
LINEAR_TASK_CLASSIFICATION = 1

MLP_TASK_CLASSIFICATION = 1

DEFAULT_BUILD_DIR = "cmake-build-release"
BUILD_DIR_ENV_VAR = "PA_ML_BUILD_DIR"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_build_dir_name() -> str:
    return os.environ.get(BUILD_DIR_ENV_VAR, DEFAULT_BUILD_DIR)


def get_library_path() -> Path:
    return get_project_root() / get_build_dir_name() / "libpa_ml.dll"


def configure_windows_dll_directories() -> None:
    if os.name != "nt":
        return

    mingw_bin = Path(r"C:\mingw64\bin")
    if mingw_bin.exists():
        os.add_dll_directory(str(mingw_bin))


def load_library() -> ctypes.CDLL:
    configure_windows_dll_directories()

    dll_path = get_library_path()
    if not dll_path.exists():
        raise FileNotFoundError(
            "Bibliothèque introuvable: "
            f"{dll_path}. "
            "Compilez la bibliothèque ou définissez "
            f"{BUILD_DIR_ENV_VAR}=<dossier-de-build>."
        )

    return ctypes.CDLL(str(dll_path))


def configure_linear_api(lib: ctypes.CDLL) -> None:
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


def configure_mlp_api(lib: ctypes.CDLL) -> None:
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

    lib.predict_mlp_model_raw.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.predict_mlp_model_raw.restype = ctypes.c_int32

    lib.destroy_mlp_model.argtypes = [ctypes.c_void_p]
    lib.destroy_mlp_model.restype = None


def configure_legacy_api(lib: ctypes.CDLL) -> None:
    lib.my_add.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.my_add.restype = ctypes.c_int32

    lib.sum_array.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    lib.sum_array.restype = ctypes.c_float


def configure_api(lib: ctypes.CDLL) -> ctypes.CDLL:
    configure_legacy_api(lib)
    configure_linear_api(lib)
    configure_mlp_api(lib)
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