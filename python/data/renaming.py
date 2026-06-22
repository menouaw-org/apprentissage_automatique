from pathlib import Path
import csv

DATA_ROOT = Path("data/raw")
CLASSES = ["dog", "cat", "others"]
VALID_EXTENSIONS = {".jpg", ".jpeg"}
OUTPUT_EXTENSION = ".jpg"
DRY_RUN = False

rows = []

for class_name in CLASSES:
    class_dir = DATA_ROOT / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {class_dir}")

    files = sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )

    for index, old_path in enumerate(files, start=1):
        new_filename = f"{class_name}_{index:05d}{OUTPUT_EXTENSION}"
        new_path = class_dir / new_filename

        rows.append(
            {
                "class": class_name,
                "old_path": str(old_path),
                "new_path": str(new_path),
                "old_filename": old_path.name,
                "new_filename": new_filename,
            }
        )

        if old_path == new_path:
            continue

        if new_path.exists():
            raise FileExistsError(f"Nom déjà utilisé: {new_path}")

        if not DRY_RUN:
            old_path.rename(new_path)

print(f"{len(rows)} fichiers analysés")
print(f"Mode simulation: {DRY_RUN}")