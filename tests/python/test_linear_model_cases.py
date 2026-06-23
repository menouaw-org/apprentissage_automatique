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
    LINEAR_TASK_CLASSIFICATION,
    LINEAR_TASK_REGRESSION,
    as_double_pointer,
    lib,
)

REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "linear_model_classic_tests.csv"

ExpectedStatus = Literal["OK", "KO"]


@dataclass
class CaseResult:
    case_name: str
    family: str
    expected: ExpectedStatus
    observed: ExpectedStatus
    status: str
    comment: str


def expected_to_status(expected: ExpectedStatus, observed: ExpectedStatus) -> str:
    if expected == "OK" and observed == "OK":
        return "validé"

    if expected == "OK" and observed == "KO":
        return "bug probable"

    if expected == "KO" and observed == "KO":
        return "limite attendue"

    return "à examiner"


def bipolar_accuracy(predictions: np.ndarray, expected: np.ndarray) -> float:
    if expected.ndim == 1:
        expected = expected.reshape(-1, 1)

    return float(np.mean(np.all(predictions == expected, axis=1)))


def predict_many(model: int, x: np.ndarray, output_size: int) -> np.ndarray:
    predictions = []

    for x_sample in x:
        y_pred = np.zeros(output_size, dtype=np.float64)
        _, x_ptr = as_double_pointer(x_sample)
        y_pred_array, y_pred_ptr = as_double_pointer(y_pred)

        predict_status = lib.predict_linear_model(model, x_ptr, y_pred_ptr)
        assert predict_status == 0, (
            f"predict_linear_model a renvoyé {predict_status}."
        )

        predictions.append(y_pred_array.copy())

    return np.array(predictions, dtype=np.float64)


def run_classification_case(
        case_name: str,
        x: np.ndarray,
        y: np.ndarray,
        expected: ExpectedStatus,
        comment: str,
        learning_rate: float = 0.1,
        epochs: int = 100,
        min_success_accuracy: float = 0.95,
) -> CaseResult:
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    input_size = int(x.shape[1])
    output_size = int(y.shape[1])

    x_array, x_ptr = as_double_pointer(x)
    _, y_ptr = as_double_pointer(y)

    model = lib.create_linear_model(
        input_size,
        output_size,
        LINEAR_TASK_CLASSIFICATION,
    )
    assert model, f"Le modèle n’a pas été créé pour le cas {case_name}."

    try:
        train_status = lib.train_linear_model(
            model,
            x_ptr,
            y_ptr,
            x_array.shape[0],
            learning_rate,
            epochs,
        )
        assert train_status == 0, (
            f"train_linear_model a renvoyé {train_status} pour {case_name}."
        )

        predictions = predict_many(model, x, output_size)
        accuracy = bipolar_accuracy(predictions, y)

    finally:
        lib.destroy_linear_model(ctypes.c_void_p(model))

    observed: ExpectedStatus = "OK" if accuracy >= min_success_accuracy else "KO"
    status = expected_to_status(expected, observed)

    return CaseResult(
        case_name=case_name,
        family="classification" if output_size == 1 else "classification_multiclass",
        expected=expected,
        observed=observed,
        status=status,
        comment=f"{comment}; accuracy={accuracy:.3f}",
    )


def run_regression_case(
        case_name: str,
        x: np.ndarray,
        y: np.ndarray,
        expected: ExpectedStatus,
        comment: str,
        learning_rate: float = 0.01,
        epochs: int = 5000,
        max_error_tolerance: float = 0.5,
) -> CaseResult:
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    input_size = int(x.shape[1])
    output_size = int(y.shape[1])

    x_array, x_ptr = as_double_pointer(x)
    _, y_ptr = as_double_pointer(y)

    model = lib.create_linear_model(
        input_size,
        output_size,
        LINEAR_TASK_REGRESSION,
    )
    assert model, f"Le modèle n’a pas été créé pour le cas {case_name}."

    try:
        train_status = lib.train_linear_model(
            model,
            x_ptr,
            y_ptr,
            x_array.shape[0],
            learning_rate,
            epochs,
        )
        assert train_status == 0, (
            f"train_linear_model a renvoyé {train_status} pour {case_name}."
        )

        predictions = predict_many(model, x, output_size)
        max_abs_error = float(np.max(np.abs(predictions - y)))

    finally:
        lib.destroy_linear_model(ctypes.c_void_p(model))

    observed: ExpectedStatus = "OK" if max_abs_error <= max_error_tolerance else "KO"
    status = expected_to_status(expected, observed)

    return CaseResult(
        case_name=case_name,
        family="regression",
        expected=expected,
        observed=observed,
        status=status,
        comment=f"{comment}; max_abs_error={max_abs_error:.3f}",
    )


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


def build_cross_case() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(44)

    x = rng.random((500, 2), dtype=np.float64) * 2.0 - 1.0
    y = np.array(
        [1.0 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1.0 for p in x],
        dtype=np.float64,
    )

    return x, y


def run_all_cases() -> list[CaseResult]:
    results: list[CaseResult] = []

    results.append(
        run_classification_case(
            case_name="Linear Simple",
            x=np.array(
                [
                    [1.0, 1.0],
                    [2.0, 3.0],
                    [3.0, 3.0],
                ],
                dtype=np.float64,
            ),
            y=np.array([1.0, -1.0, -1.0], dtype=np.float64),
            expected="OK",
            comment="cas linéaire simple du professeur",
            epochs=50,
        )
    )

    rng = np.random.default_rng(45)
    x_linear_multiple = np.concatenate(
        [
            rng.random((50, 2), dtype=np.float64) * 0.9 + np.array([1.0, 1.0]),
            rng.random((50, 2), dtype=np.float64) * 0.9 + np.array([2.0, 2.0]),
            ]
    )
    y_linear_multiple = np.concatenate(
        [
            np.ones(50, dtype=np.float64),
            np.ones(50, dtype=np.float64) * -1.0,
            ]
    )

    results.append(
        run_classification_case(
            case_name="Linear Multiple",
            x=x_linear_multiple,
            y=y_linear_multiple,
            expected="OK",
            comment="cas linéaire multiple du professeur",
            epochs=100,
        )
    )

    results.append(
        run_classification_case(
            case_name="XOR",
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
            expected="KO",
            comment="limite non linéaire attendue",
            epochs=100,
        )
    )

    x_cross, y_cross = build_cross_case()
    results.append(
        run_classification_case(
            case_name="Cross",
            x=x_cross,
            y=y_cross,
            expected="KO",
            comment="limite non linéaire attendue",
            epochs=100,
        )
    )

    x_multi_linear, y_multi_linear = build_multi_linear_3_classes_case()
    results.append(
        run_classification_case(
            case_name="Multi Linear 3 classes",
            x=x_multi_linear,
            y=y_multi_linear,
            expected="OK",
            comment="cas prioritaire pour le projet",
            epochs=200,
            min_success_accuracy=0.90,
        )
    )

    x_multi_cross, y_multi_cross = build_multi_cross_case()
    results.append(
        run_classification_case(
            case_name="Multi Cross",
            x=x_multi_cross,
            y=y_multi_cross,
            expected="KO",
            comment="limite non linéaire multi-classes attendue",
            epochs=200,
            min_success_accuracy=0.90,
        )
    )

    results.append(
        run_regression_case(
            case_name="Linear Simple 2D",
            x=np.array([[1.0], [2.0]], dtype=np.float64),
            y=np.array([2.0, 3.0], dtype=np.float64),
            expected="OK",
            comment="cas de régression linéaire simple",
            max_error_tolerance=0.2,
        )
    )

    results.append(
        run_regression_case(
            case_name="Non Linear Simple 2D",
            x=np.array([[1.0], [2.0], [3.0]], dtype=np.float64),
            y=np.array([2.0, 3.0, 2.5], dtype=np.float64),
            expected="OK",
            comment="cas de régression traité comme diagnostic d’approximation",
            max_error_tolerance=0.6,
        )
    )

    results.append(
        run_regression_case(
            case_name="Linear Simple 3D",
            x=np.array(
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 1.0],
                ],
                dtype=np.float64,
            ),
            y=np.array([2.0, 3.0, 2.5], dtype=np.float64),
            expected="OK",
            comment="cas de régression linéaire à deux entrées",
            max_error_tolerance=0.3,
        )
    )

    results.append(
        run_regression_case(
            case_name="Linear Tricky 3D",
            x=np.array(
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                ],
                dtype=np.float64,
            ),
            y=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            expected="OK",
            comment="cas linéaire colinéaire du professeur",
            max_error_tolerance=0.3,
        )
    )

    results.append(
        run_regression_case(
            case_name="Non Linear Simple 3D",
            x=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            y=np.array([2.0, 1.0, -2.0, -1.0], dtype=np.float64),
            expected="KO",
            comment="limite non linéaire attendue en régression",
            max_error_tolerance=0.5,
        )
    )

    return results


def write_report(results: list[CaseResult]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "case_name",
            "family",
            "expected",
            "observed",
            "status",
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
                    "comment": result.comment,
                }
            )


def main() -> None:
    results = run_all_cases()
    write_report(results)

    for result in results:
        print(
            f"{result.case_name}: expected={result.expected}, "
            f"observed={result.observed}, status={result.status}"
        )

    blocking_results = [
        result
        for result in results
        if result.status in {"bug probable", "à examiner"}
    ]

    if blocking_results:
        details = ", ".join(result.case_name for result in blocking_results)
        raise AssertionError(
            "Certains cas professeur nécessitent une analyse: "
            f"{details}. Voir {REPORT_PATH}."
        )

    print(f"Rapport écrit dans: {REPORT_PATH}")
    print("Validation des cas professeur terminée.")


if __name__ == "__main__":
    main()