from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

RAW_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")

CLASSES = ["dog", "cat", "others"]
SIZES = [32, 64, 128]

INPUT_EXTENSION = ".jpg"
OUTPUT_EXTENSION = ".jpg"
JPEG_QUALITY = 95
BACKGROUND_COLOR = (0, 0, 0)


def resize_with_padding(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")

    resized = ImageOps.contain(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS,
    )

    output = Image.new("RGB", (size, size), BACKGROUND_COLOR)

    x = (size - resized.width) // 2
    y = (size - resized.height) // 2

    output.paste(resized, (x, y))

    return output


def process_class(class_name: str, size: int) -> int:
    source_dir = RAW_ROOT / class_name
    target_dir = PROCESSED_ROOT / f"{size}x{size}" / class_name

    if not source_dir.exists():
        raise FileNotFoundError(f"Dossier source introuvable: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() == INPUT_EXTENSION
    )

    processed_count = 0
    progress_label = f"{size}x{size} / {class_name}"

    for image_path in tqdm(image_paths, desc=progress_label, unit="image"):
        target_path = target_dir / f"{image_path.stem}{OUTPUT_EXTENSION}"

        try:
            with Image.open(image_path) as image:
                processed_image = resize_with_padding(image, size)
                processed_image.save(
                    target_path,
                    format="JPEG",
                    quality=JPEG_QUALITY,
                    optimize=True,
                )

            processed_count += 1

        except UnidentifiedImageError:
            tqdm.write(f"Image illisible ignorée: {image_path}")

    return processed_count


def main() -> None:
    total_processed_count = 0

    for size in SIZES:
        for class_name in CLASSES:
            count = process_class(class_name, size)
            total_processed_count += count

    print(f"\nTotal: {total_processed_count} images traitées")


if __name__ == "__main__":
    main()