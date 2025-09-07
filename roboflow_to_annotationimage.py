import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rich import print

CATEGORY_COLORS = {
    0: (255, 255, 255),  # White - Wire/Nozzle
    1: (0, 255, 0),  # Green - Material
}


def return_directory_with_roboflow_info(root_dir):
    info_dir = root_dir / "train"
    if not info_dir.exists():
        print(f"Error: {root_dir} does not exist")
        return
    if not info_dir.is_dir():
        print(f"Error: {info_dir} is not a directory")
        return
    return info_dir


def process_coco_json_to_create_annotation_images(info_dir: Path):
    """Process COCO format JSON file to create annotation images"""
    coco_json_path = info_dir / "_annotations.coco.json"
    assert coco_json_path.exists(), f"Error: {coco_json_path} does not exist"

    annotation_dir = coco_json_path.parent.parent / "Annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Create lookup dictionaries
    images_dict = {img["id"]: img for img in coco_data["images"]}
    categories_dict = {cat["id"]: cat for cat in coco_data["categories"]}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    for image_id, annotations in annotations_by_image.items():
        image_info = images_dict[image_id]
        print(
            f"[bold yellow]Processing image {image_id}: {image_info['file_name']}[/bold yellow]"
        )

        # Extract original filename without extension for output
        original_name = (
            image_info["extra"]["name"]
            if "extra" in image_info
            else image_info["file_name"]
        )
        base_name = Path(original_name).stem
        output_path = annotation_dir / f"{base_name}.png"

        height, width = image_info["height"], image_info["width"]
        mask_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Sort annotations so that categories containing "wire" or "nozzle" appear on top
        def get_sort_key(ann):
            category_id = ann["category_id"]
            category_name = categories_dict[category_id]["name"].lower()
            # Return 0 for wire/nozzle categories (top priority), 1 for others
            return 0 if ("wire" in category_name or "nozzle" in category_name) else 1

        sorted_annotations = sorted(annotations, key=get_sort_key, reverse=True)

        # Process each annotation
        for ann in sorted_annotations:
            if any(
                category_name.lower()
                in categories_dict[ann["category_id"]]["name"].lower()
                for category_name in ["wire", "nozzle"]
            ):
                color = CATEGORY_COLORS[0]
            else:
                color = CATEGORY_COLORS[1]

            # Convert polygon segmentation to mask
            seg = ann["segmentation"]
            if seg:  # Check if segmentation exists
                # seg is a list of polygons, each polygon is a list of coordinates
                for polygon in seg:
                    if len(polygon) >= 6:  # Need at least 3 points (6 coordinates)
                        # Convert to numpy array and reshape
                        polygon_array = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                        # Fill the polygon with the category color
                        cv2.fillPoly(mask_image, [polygon_array], color)

        cv2.imwrite(str(output_path), mask_image)
        print(f"[bold green]Saved {output_path}[/bold green]")

    print(
        f"[bold blue]Finished COCO Annotation Conversion for {coco_json_path}[/bold blue]"
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python roboflow_to_annotationimage.py <datasetsdir>")
        sys.exit(1)
    datasets_dir_path = Path(sys.argv[1])
    assert datasets_dir_path.exists(), f"Error: {datasets_dir_path} does not exist"
    assert datasets_dir_path.is_dir(), f"Error: {datasets_dir_path} is not a directory"
    for dataset_dir in datasets_dir_path.glob("*"):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        info_dir_path = return_directory_with_roboflow_info(dataset_dir)
        process_coco_json_to_create_annotation_images(info_dir_path)


if __name__ == "__main__":
    main()
