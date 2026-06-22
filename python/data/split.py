from pathlib import Path
import csv
import random
from collections import defaultdict

DATA_ROOT = Path("data/processed/64x64")
SPLITS_ROOT = Path("data/splits")

CLASSES = ["dog", "cat", "others"]
IMAGE_EXTENSION = ".jpg"

TEST_RATIO = 0.15
N_FOLDS = 5
RANDOM_SEED = 42

TEST_CSV = SPLITS_ROOT / "test.csv"
FOLDS_CSV = SPLITS_ROOT / "folds.csv"


def collect_images() -> dict[str, list[Path]]:
    images_by_class = {}

    for class_name in CLASSES:
        class_dir = DATA_ROOT / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Dossier introuvable: {class_dir}")

        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() == IMAGE_EXTENSION
        )

        if not image_paths:
            raise ValueError(f"Aucune image trouvée pour la classe: {class_name}")

        images_by_class[class_name] = image_paths

    return images_by_class


def relative_path(path: Path) -> str:
    return path.as_posix()


def build_stratified_splits(images_by_class: dict[str, list[Path]]):
    rng = random.Random(RANDOM_SEED)

    test_rows = []
    fold_rows = []

    for class_name, image_paths in images_by_class.items():
        shuffled_paths = image_paths[:]
        rng.shuffle(shuffled_paths)

        test_count = round(len(shuffled_paths) * TEST_RATIO)

        test_paths = shuffled_paths[:test_count]
        cross_validation_paths = shuffled_paths[test_count:]

        for path in test_paths:
            test_rows.append(
                {
                    "path": relative_path(path),
                    "label": class_name,
                    "split": "test",
                }
            )

        for index, path in enumerate(cross_validation_paths):
            fold = index % N_FOLDS

            fold_rows.append(
                {
                    "path": relative_path(path),
                    "label": class_name,
                    "fold": fold,
                }
            )

    return test_rows, fold_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], group_key: str) -> None:
    counts = defaultdict(int)

    for row in rows:
        counts[(row["label"], row[group_key])] += 1

    for key, count in sorted(counts.items()):
        label, group = key
        print(f"{label} / {group}: {count}")


def main() -> None:
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

    images_by_class = collect_images()
    test_rows, fold_rows = build_stratified_splits(images_by_class)

    test_rows = sorted(test_rows, key=lambda row: (row["label"], row["path"]))
    fold_rows = sorted(fold_rows, key=lambda row: (row["fold"], row["label"], row["path"]))

    write_csv(TEST_CSV, test_rows, ["path", "label", "split"])
    write_csv(FOLDS_CSV, fold_rows, ["path", "label", "fold"])

    print(f"Fichier généré: {TEST_CSV}")
    print(f"Fichier généré: {FOLDS_CSV}")

    print("\nRépartition du test final:")
    summarize(test_rows, "split")

    print("\nRépartition des plis:")
    summarize(fold_rows, "fold")


if __name__ == "__main__":
    main()