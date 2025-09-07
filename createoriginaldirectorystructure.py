import shutil
from pathlib import Path


def create_original_directory_structure(
    roboflow_annotations_dir: Path,
    root_save_dir: Path,
    original_images_dir: Path,
):
    copyfrom_annotations_dir = roboflow_annotations_dir / "Annotations"

    JPGImages_dir = root_save_dir / roboflow_annotations_dir.stem / "JPGImages"
    JPGImages_dir.mkdir(parents=True, exist_ok=True)
    Annotations_dir = root_save_dir / roboflow_annotations_dir.stem / "Annotations"
    Annotations_dir.mkdir(parents=True, exist_ok=True)
    for dir_path, dirs, files in Path(original_images_dir).walk():
        files = list(filter(lambda x: not str(x).startswith("."), files))
        new_images_dir = JPGImages_dir / dir_path.relative_to(original_images_dir)
        new_annotations_dir = Annotations_dir / dir_path.relative_to(
            original_images_dir
        )
        new_images_dir.mkdir(parents=True, exist_ok=True)
        new_annotations_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            file = Path(file)
            shutil.copy(dir_path / file, new_images_dir / file)
            assert (copyfrom_annotations_dir / (file.stem + ".png")).exists(), (
                f"Error: {copyfrom_annotations_dir / (file.stem + '.png')} does not exist"
            )
            shutil.copy(
                copyfrom_annotations_dir / (file.stem + ".png"),
                new_annotations_dir / (file.stem + ".png"),
            )

    # Convert from JPEG to JPG
    for dir_path, dirs, files in Path(root_save_dir).walk():
        files = list(sorted(filter(lambda x: not str(x).startswith("."), files)))
        for idx, file in enumerate(files):
            file = Path(file)
            if file.suffix.lower() in [".jpeg", ".jpg"]:
                new_file_name = dir_path / (f"{idx:05d}" + ".jpg")
            else:
                new_file_name = dir_path / (f"{idx:05d}" + file.suffix)
            (dir_path / file).rename(new_file_name)


def main():
    roboflow_dirs = list(
        sorted(filter(lambda x: x.is_dir(), Path("roboflowdata").iterdir()))
    )
    original_dirs = [
        "/Users/calwetzel/Downloads/WAAMlabeledDataset/Roboflow_SAM2_Frames",
        "/Volumes/Extreme SSD/ORNL/TIG/dataset",
        "/Users/calwetzel/Desktop/COSC/ORNL/SCOPSdata/irPOLYMER",
        "/Users/calwetzel/Desktop/COSC/ORNL/SCOPSdata/visPOLYMER",
    ]
    for roboflow_dir, original_dir in zip(roboflow_dirs, original_dirs):
        create_original_directory_structure(
            Path(roboflow_dir),
            Path("./SAM2images"),
            Path(original_dir),
        )


if __name__ == "__main__":
    main()
