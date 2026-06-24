import csv
import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "mlp_model_classic_tests.csv"

ExpectedStatus = Literal["OK", "OK à confirmer"]


@dataclass
class CaseConfig:
    case_name: str
    family: str
    x: np.ndarray
    y: np.ndarray
    hidden_sizes: np.ndarray
    learning_rate: float
    epochs: int
    expected: ExpectedStatus
    min_accuracy: float
    comment: str


@dataclass
class CaseResult:
    case_name: str
    family: str
    expected: str
    observed: str
    status: str
    accuracy: float
    architecture: str
    comment: str


def architecture_label(input_size: int, hidden_sizes: np.ndarray, output_size: int) -> str:
    sizes = [str(input_size)]
    sizes.extend(str(int(size)) for size in hidden_sizes)
    sizes.append(str(output_size))
    return " -> ".join(sizes)


def expected_to_status(expected: str, observed: str) -> str:
    if expected == "OK" and observed == "OK":
        return "validé"

    if expected == "OK" and observed == "KO":
        return "bug probable ou réglage insuffisant"

    if expected == "OK à confirmer" and observed == "OK":
        return "validé"

    return "à analyser"


def predict_one(model: int, x_sample: np.ndarray, output_size: int) -> np.ndarray:
    y_pred = np.zeros(output_size, dtype=np.float64)

    _, x_ptr = as_double_pointer(x_sample)
    y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

    predict_status = lib.predict_mlp_model(model, x_ptr, y_pred_ptr)
    assert predict_status == 0, f"predict_mlp_model a renvoyé {predict_status}."

    return y_pred_array.copy()


def predict_many(model: int, x: np.ndarray, output_size: int) -> np.ndarray:
    predictions = [
        predict_one(model, x_sample, output_size)
        for x_sample in x
    ]

    return np.asarray(predictions, dtype=np.float64)


def bipolar_accuracy(predictions: np.ndarray, expected: np.ndarray) -> float:
    if expected.ndim == 1:
        expected = expected.reshape(-1, 1)

    return float(np.mean(np.all(predictions == expected, axis=1)))


def run_case(config: CaseConfig) -> CaseResult:
    x = np.ascontiguousarray(config.x, dtype=np.float64)
    y = np.ascontiguousarray(config.y, dtype=np.float64)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    hidden_sizes = np.ascontiguousarray(config.hidden_sizes, dtype=np.int32)

    input_size = int(x.shape[1])
    output_size = int(y.shape[1])
    hidden_layer_count = int(hidden_sizes.shape[0])

    x_array, x_ptr = as_double_pointer(x)
    _, y_ptr = as_double_pointer(y)
    hidden_sizes_array, hidden_sizes_ptr = as_int32_pointer(hidden_sizes)

    model = lib.create_mlp_model(
        input_size,
        output_size,
        hidden_layer_count,
        hidden_sizes_ptr,
        MLP_TASK_CLASSIFICATION,
    )
    assert model, f"Le modèle MLP n’a pas été créé pour {config.case_name}."

    try:
        train_status = lib.train_mlp_model(
            model,
            x_ptr,
            y_ptr,
            x_array.shape[0],
            config.learning_rate,
            config.epochs,
        )
        assert train_status == 0, (
            f"train_mlp_model a renvoyé {train_status} pour {config.case_name}."
        )

        predictions = predict_many(model, x, output_size)
        accuracy = bipolar_accuracy(predictions, y)

    finally:
        lib.destroy_mlp_model(ctypes.c_void_p(model))

    observed = "OK" if accuracy >= config.min_accuracy else "KO"

    return CaseResult(
        case_name=config.case_name,
        family=config.family,
        expected=config.expected,
        observed=observed,
        status=expected_to_status(config.expected, observed),
        accuracy=accuracy,
        architecture=architecture_label(input_size, hidden_sizes_array, output_size),
        comment=config.comment,
    )


def build_linear_multiple_case() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(45)

    x = np.concatenate(
        [
            rng.random((50, 2), dtype=np.float64) * 0.9 + np.array([1.0, 1.0]),
            rng.random((50, 2), dtype=np.float64) * 0.9 + np.array([2.0, 2.0]),
            ]
    )
    y = np.concatenate(
        [
            np.ones(50, dtype=np.float64),
            np.ones(50, dtype=np.float64) * -1.0,
            ]
    )

    return x, y


def build_cross_case() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(44)

    x = rng.random((500, 2), dtype=np.float64) * 2.0 - 1.0
    y = np.array(
        [1.0 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1.0 for p in x],
        dtype=np.float64,
    )

    return x, y


def build_multi_linear_3_classes_case() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    x = rng.random((500, 2), dtype=np.float64) * 2.0 - 1.0
    y = np.array(
        [
            [1.0, -1.0, -1.0]
            if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0
            else [-1.0, 1.0, -1.0]
            if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0
            else [-1.0, -1.0, 1.0]
            if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0
            else [-1.0, -1.0, -1.0]
            for p in x
        ],
        dtype=np.float64,
    )

    mask = ~np.all(y == [-1.0, -1.0, -1.0], axis=1)

    return x[mask], y[mask]


def build_multi_cross_case() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(43)

    x = rng.random((1000, 2), dtype=np.float64) * 2.0 - 1.0
    y = np.array(
        [
            [1.0, -1.0, -1.0]
            if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25
            else [-1.0, 1.0, -1.0]
            if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25
            else [-1.0, -1.0, 1.0]
            for p in x
        ],
        dtype=np.float64,
    )

    return x, y


def build_cases() -> list[CaseConfig]:
    x_linear_multiple, y_linear_multiple = build_linear_multiple_case()
    x_cross, y_cross = build_cross_case()
    x_multi_linear, y_multi_linear = build_multi_linear_3_classes_case()
    x_multi_cross, y_multi_cross = build_multi_cross_case()

    return [
        CaseConfig(
            case_name="Linear Simple",
            family="classification",
            x=np.array(
                [
                    [1.0, 1.0],
                    [2.0, 3.0],
                    [3.0, 3.0],
                ],
                dtype=np.float64,
            ),
            y=np.array([1.0, -1.0, -1.0], dtype=np.float64),
            hidden_sizes=np.array([2], dtype=np.int32),
            learning_rate=0.05,
            epochs=1000,
            expected="OK",
            min_accuracy=1.0,
            comment="cas linéaire simple du professeur",
        ),
        CaseConfig(
            case_name="Linear Multiple",
            family="classification",
            x=x_linear_multiple,
            y=y_linear_multiple,
            hidden_sizes=np.array([2], dtype=np.int32),
            learning_rate=0.05,
            epochs=1000,
            expected="OK",
            min_accuracy=0.95,
            comment="cas linéaire multiple du professeur",
        ),
        CaseConfig(
            case_name="XOR",
            family="classification_non_linear",
            x=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                ],
                dtype=np.float64,
            ),
            y=np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64),
            hidden_sizes=np.array([4], dtype=np.int32),
            learning_rate=0.03,
            epochs=10000,
            expected="OK",
            min_accuracy=1.0,
            comment="preuve minimale de non-linéarité",
        ),
        CaseConfig(
            case_name="Cross",
            family="classification_non_linear",
            x=x_cross,
            y=y_cross,
            hidden_sizes=np.array([8], dtype=np.int32),
            learning_rate=0.03,
            epochs=2000,
            expected="OK",
            min_accuracy=0.90,
            comment="cas non linéaire dense du professeur",
        ),
        CaseConfig(
            case_name="Multi Linear 3 classes",
            family="classification_multiclass",
            x=x_multi_linear,
            y=y_multi_linear,
            hidden_sizes=np.array([4], dtype=np.int32),
            learning_rate=0.05,
            epochs=2000,
            expected="OK",
            min_accuracy=0.90,
            comment="cas multi-classes le plus proche du périmètre dog/cat/others",
        ),
        CaseConfig(
            case_name="Multi Cross",
            family="classification_multiclass_non_linear",
            x=x_multi_cross,
            y=y_multi_cross,
            hidden_sizes=np.array([8, 6], dtype=np.int32),
            learning_rate=0.03,
            epochs=3000,
            expected="OK à confirmer",
            min_accuracy=0.80,
            comment="cas multi-classes non linéaire; architecture à ajuster si nécessaire",
        ),
    ]


def write_report(results: list[CaseResult]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "case_name",
            "family",
            "expected",
            "observed",
            "status",
            "accuracy",
            "architecture",
            "comment",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "case_name": result.case_name,
                    "family": result.family,
                    "expected": result.expected,
                    "observed": result.observed,
                    "status": result.status,
                    "accuracy": f"{result.accuracy:.4f}",
                    "architecture": result.architecture,
                    "comment": result.comment,
                }
            )


def main() -> None:
    results = [run_case(config) for config in build_cases()]
    write_report(results)

    for result in results:
        print(
            f"{result.case_name}: expected={result.expected}, "
            f"observed={result.observed}, accuracy={result.accuracy:.4f}, "
            f"status={result.status}"
        )

    blocking_results = [
        result
        for result in results
        if result.expected == "OK" and result.observed != "OK"
    ]

    if blocking_results:
        details = ", ".join(result.case_name for result in blocking_results)
        raise AssertionError(
            "Certains cas MLP attendus OK nécessitent une analyse: "
            f"{details}. Voir {REPORT_PATH}."
        )

    print(f"Rapport écrit dans: {REPORT_PATH}")
    print("Validation des cas MLP terminée.")


if __name__ == "__main__":
    main()