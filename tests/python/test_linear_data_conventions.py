import ctypes
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.bindings.c_api import (
    LINEAR_TASK_REGRESSION,
    as_double_pointer,
    lib,
)


def test_linear_regression_two_inputs() -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )

    y = np.array(
        [
            [1.0],
            [3.0],
            [-2.0],
            [0.0],
            [2.0],
            [-3.0],
        ],
        dtype=np.float64,
    )

    x_test = np.array([2.0, 3.0], dtype=np.float64)
    y_pred = np.zeros(1, dtype=np.float64)

    x_array, x_ptr = as_double_pointer(x)
    _, y_ptr = as_double_pointer(y)
    _, x_test_ptr = as_double_pointer(x_test)
    y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

    model = lib.create_linear_model(2, 1, LINEAR_TASK_REGRESSION)
    assert model, "Le modèle linéaire n’a pas été créé."

    try:
        train_status = lib.train_linear_model(
            model,
            x_ptr,
            y_ptr,
            x_array.shape[0],
            0.01,
            5000,
        )
        assert train_status == 0

        predict_status = lib.predict_linear_model(model, x_test_ptr, y_pred_ptr)
        assert predict_status == 0

        expected = 2.0 * 2.0 - 3.0 * 3.0 + 1.0
        prediction = float(y_pred_array[0])

        assert abs(prediction - expected) < 1.0

    finally:
        lib.destroy_linear_model(ctypes.c_void_p(model))


if __name__ == "__main__":
    test_linear_regression_two_inputs()
    print("Test de conventions de données réussi.")