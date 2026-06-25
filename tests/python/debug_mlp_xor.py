import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.bindings.c_api import (
    MLP_TASK_CLASSIFICATION,
    as_double_pointer,
    as_int32_pointer,
    lib,
)


@dataclass
class DebugConfig:
    name: str
    x: np.ndarray
    y: np.ndarray
    hidden_sizes: np.ndarray
    learning_rate: float
    epochs: int


def predict_one(model: int, x_sample: np.ndarray) -> np.ndarray:
    y_pred = np.zeros(1, dtype=np.float64)

    _, x_ptr = as_double_pointer(x_sample)
    y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

    predict_status = lib.predict_mlp_model(model, x_ptr, y_pred_ptr)
    assert predict_status == 0, f"predict_mlp_model a renvoyé {predict_status}."

    return y_pred_array.copy()


def run_config(config: DebugConfig) -> None:
    x = np.ascontiguousarray(config.x, dtype=np.float64)
    y = np.ascontiguousarray(config.y.reshape(-1, 1), dtype=np.float64)
    hidden_sizes = np.ascontiguousarray(config.hidden_sizes, dtype=np.int32)

    x_array, x_ptr = as_double_pointer(x)
    _, y_ptr = as_double_pointer(y)
    hidden_sizes_array, hidden_sizes_ptr = as_int32_pointer(hidden_sizes)

    model = lib.create_mlp_model(
        int(x.shape[1]),
        1,
        int(hidden_sizes_array.shape[0]),
        hidden_sizes_ptr,
        MLP_TASK_CLASSIFICATION,
    )
    assert model, f"Le modèle MLP n’a pas été créé pour {config.name}."

    try:
        train_status = lib.train_mlp_model(
            model,
            x_ptr,
            y_ptr,
            x_array.shape[0],
            config.learning_rate,
            config.epochs,
        )
        assert train_status == 0, f"train_mlp_model a renvoyé {train_status}."

        predictions = np.asarray([predict_one(model, row) for row in x])
        accuracy = float(np.mean(predictions.reshape(-1) == config.y.reshape(-1)))

        print()
        print("=" * 80)
        print(config.name)
        print(f"hidden_sizes={hidden_sizes_array.tolist()}")
        print(f"learning_rate={config.learning_rate}")
        print(f"epochs={config.epochs}")
        print(f"accuracy={accuracy:.4f}")
        print("points:")

        for x_sample, expected, predicted in zip(
                x,
                config.y.reshape(-1),
                predictions.reshape(-1),
        ):
            print(
                f"  x={x_sample.tolist()} "
                f"expected={expected:+.1f} "
                f"predicted={predicted:+.1f}"
            )

    finally:
        lib.destroy_mlp_model(ctypes.c_void_p(model))


def main() -> None:
    x_original = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)

    x_centered = np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    configs = [
        DebugConfig(
            name="xor_centered_hidden_4_lr_0.03_epochs_20000",
            x=x_centered,
            y=y,
            hidden_sizes=np.array([4], dtype=np.int32),
            learning_rate=0.03,
            epochs=20000,
        )
    ]

    for x_name, x_values in [
        ("xor_original_0_1", x_original),
        ("xor_centered_minus_1_plus_1", x_centered),
    ]:
        for hidden in [[2], [4], [8]]:
            for learning_rate in [0.01, 0.03, 0.05]:
                for epochs in [5000, 10000, 20000]:
                    configs.append(
                        DebugConfig(
                            name=(
                                f"{x_name}_hidden_{hidden[0]}_"
                                f"lr_{learning_rate}_epochs_{epochs}"
                            ),
                            x=x_values,
                            y=y,
                            hidden_sizes=np.array(hidden, dtype=np.int32),
                            learning_rate=learning_rate,
                            epochs=epochs,
                        )
                    )

    for config in configs:
        run_config(config)


if __name__ == "__main__":
    main()