import ctypes
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.bindings.c_api import (
    LINEAR_TASK_CLASSIFICATION,
    as_double_pointer,
    lib,
)


def test_multiclass_prediction_outputs_bipolar_vector() -> None:
    input_size = 2
    output_size = 3

    x_test = np.array([1.0, -1.0], dtype=np.float64)
    y_pred = np.zeros(output_size, dtype=np.float64)

    _, x_test_ptr = as_double_pointer(x_test)
    y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

    model = lib.create_linear_model(
        input_size,
        output_size,
        LINEAR_TASK_CLASSIFICATION,
    )
    assert model, "Le modèle linéaire multi-sorties n'a pas été créé."

    try:
        predict_status = lib.predict_linear_model(model, x_test_ptr, y_pred_ptr)
        assert predict_status == 0

        expected = np.array([1.0, -1.0, -1.0], dtype=np.float64)
        np.testing.assert_array_equal(y_pred_array, expected)

    finally:
        lib.destroy_linear_model(ctypes.c_void_p(model))


if __name__ == "__main__":
    test_multiclass_prediction_outputs_bipolar_vector()
    print("Test de prédiction multi-sorties réussi.")