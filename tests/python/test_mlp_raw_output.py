import ctypes
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.bindings.c_api import (  # noqa: E402
    MLP_TASK_CLASSIFICATION,
    as_double_pointer,
    as_int32_pointer,
    lib,
)


def main() -> None:
    x_sample = np.array([0.25, -0.75], dtype=np.float64)
    y_raw = np.zeros(3, dtype=np.float64)
    y_pred = np.zeros(3, dtype=np.float64)
    hidden_sizes = np.array([4], dtype=np.int32)

    _, x_ptr = as_double_pointer(x_sample)
    y_raw_array, y_raw_ptr = as_double_pointer(y_raw)
    y_pred_array, y_pred_ptr = as_double_pointer(y_pred)
    hidden_sizes_array, hidden_sizes_ptr = as_int32_pointer(hidden_sizes)

    model = lib.create_mlp_model(
        2,
        3,
        int(hidden_sizes_array.shape[0]),
        hidden_sizes_ptr,
        MLP_TASK_CLASSIFICATION,
    )
    assert model, "Le modèle MLP n’a pas été créé."

    try:
        raw_status = lib.predict_mlp_model_raw(model, x_ptr, y_raw_ptr)
        assert raw_status == 0, f"predict_mlp_model_raw a renvoyé {raw_status}."

        pred_status = lib.predict_mlp_model(model, x_ptr, y_pred_ptr)
        assert pred_status == 0, f"predict_mlp_model a renvoyé {pred_status}."

        assert y_raw_array.shape == (3,)
        assert y_pred_array.shape == (3,)

        assert np.all(y_raw_array >= -1.0)
        assert np.all(y_raw_array <= 1.0)

        assert set(np.unique(y_pred_array)).issubset({-1.0, 1.0})
        assert np.sum(y_pred_array == 1.0) == 1
        assert np.sum(y_pred_array == -1.0) == 2

        assert not np.all(np.isin(y_raw_array, [-1.0, 1.0])), (
            "La sortie raw semble déjà binarisée; vérifier que "
            "predict_mlp_model_raw copie les activations avant décision."
        )

    finally:
        lib.destroy_mlp_model(ctypes.c_void_p(model))

    print("Test de sortie continue MLP réussi.")


if __name__ == "__main__":
    main()