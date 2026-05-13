#!/usr/bin/env python3
"""Embed authorship + dataset metadata into every image of AMVOS.

Run from the repo root:
    python tag_dataset_metadata.py --root AMVOS            # write
    python tag_dataset_metadata.py --root AMVOS --dry-run  # preview

Modifies files in place. Pre-existing EXIF (JPEG) and text chunks (PNG) are
preserved; new fields are merged in.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image, PngImagePlugin
    import piexif
    from piexif.helper import UserComment
except ImportError as e:
    sys.exit(
        f"missing dependency: {e.name}\ninstall with: uv pip install Pillow piexif"
    )

AUTHORS = [
    "Calvin Wetzel",
    "Hector Santos-Villalobos",
    "James Haley",
    "Vladimir Orlyanchik",
    "Mario Rodriguez Parra",
    "Mithulan Paramanathan",
    "Tom Feldhausen",
    "Michael Sebok",
    "Chris Masuo",
    "Vincent Paquit",
]
AUTHORS_STR = "; ".join(AUTHORS)
LOCATION = "Oak Ridge National Laboratory Manufacturing Demonstration Facility (MDF), Oak Ridge, TN, USA"
COLLECTION = "AMVOS (Additive Manufacturing Video Object Segmentation)"
COPYRIGHT = f"(C) ORNL MDF -- {AUTHORS_STR}"
SOFTWARE = "tag_dataset_metadata.py v3"

# White (255,255,255) and Green (0,255,0) class semantics per dataset.
DATASETS: dict[str, dict[str, str]] = {
    "TIG": {"(255,255,255)": "Feed Wire", "(0,255,0)": "Melt Pool"},
    "PAW": {"(255,255,255)": "Feed Wire", "(0,255,0)": "Melt Pool"},
    "LHWDED": {"(255,255,255)": "Feed Wire", "(0,255,0)": "Melt Pool"},
    "irPOLYMER": {"(255,255,255)": "Nozzle", "(0,255,0)": "Material"},
    "visPOLYMER": {"(255,255,255)": "Nozzle", "(0,255,0)": "Material"},
}

# Per-dataset capture facts.
CAMERAS: dict[str, dict] = {
    "TIG": {
        "make": "Basler", "model": "Ace",
        "fps": 25, "resolution": "800x600",
        "modality": "visible",
        "mount": "external tripod, side view",
    },
    "PAW": {
        "make": "Basler", "model": "Ace",
        "fps": 25, "resolution": "800x600",
        "modality": "visible",
        "mount": "external tripod, side view",
    },
    "LHWDED": {
        "make": "Xiris", "model": "XVC-1000",
        "fps": 30, "resolution": "1280x1020",
        "modality": "visible",
        "mount": "mounted to extruder head, overview perspective",
    },
    "visPOLYMER": {
        "make": "Intel", "model": "RealSense D435i",
        "fps": 1, "resolution": "640x480",
        "modality": "visible color (RGB)",
        "mount": "integrated on HADDY 6-axis robotic arm",
    },
    "irPOLYMER": {
        "make": "FLIR", "model": "Lepton 3.5",
        "fps": 9, "resolution": "160x120",
        "modality": "infrared (thermal, up to 400 C)",
        "mount": "integrated on HADDY 6-axis robotic arm; one of four Lepton 3.5 cameras",
    },
}

PROCESS: dict[str, dict] = {
    "TIG": {
        "name": "TIG-WAAM (Tungsten Inert Gas Wire-Arc Additive Manufacturing)",
        "description": (
            "Feed wire fed into an electric arc between a non-consumable tungsten "
            "electrode and the welding surface; arc generates a melt pool that "
            "solidifies into deposited layers."
        ),
        "equipment": (
            "Lincoln Electric R450 Robotic Power Source; AutoDrive 4R220 Wire Drive; "
            "Power Wave Advanced Process Module; Binzel ABITIG WH 400W robotic GTAW "
            "torch with cold-wire feeding module; ABB IRB 2600 robot arm"
        ),
    },
    "PAW": {
        "name": "PAW (Plasma Arc Welding) wire-arc additive manufacturing",
        "description": (
            "Non-consumable tungsten electrode generates an ionized plasma stream "
            "from Argon gas, focused by a chilled copper nozzle; plasma melts the "
            "feed wire onto the substrate."
        ),
        "equipment": (
            "SBI PMI 400 TL3 plasma power supply; WF-1 wire feeder; TP450 plasma torch"
        ),
    },
    "LHWDED": {
        "name": "LHW-DED (Laser Hot-Wire Directed Energy Deposition)",
        "description": (
            "Laser melts a continuously fed wire to generate a melt pool that "
            "solidifies layer by layer. The feed wire is preheated by resistive "
            "heating before entering the melt pool."
        ),
        "equipment": (
            "Mazak VTC-800G SR AM (5-axis deposition + 5-axis machining hybrid "
            "manufacturing platform)"
        ),
    },
    "visPOLYMER": {
        "name": "Polymer pellet screw extrusion (visible imaging)",
        "description": (
            "Composite feedstock heated and extruded through a nozzle, fused to "
            "previously deposited layers. Visible imaging of nozzle, extruded "
            "material, and prior layers in a 10x10 cm region centered at the "
            "nozzle tip."
        ),
        "equipment": (
            "Custom 6-axis HADDY robotic arm with integrated extrusion and camera systems"
        ),
    },
    "irPOLYMER": {
        "name": "Polymer pellet screw extrusion (thermal/IR imaging)",
        "description": (
            "Composite feedstock heated and extruded through a nozzle, fused to "
            "previously deposited layers. Thermal/IR imaging (one of four FLIR "
            "Lepton 3.5 cameras) of nozzle, extruded material, and prior layers "
            "in a 10x10 cm region centered at the nozzle tip."
        ),
        "equipment": (
            "Custom 6-axis HADDY robotic arm with integrated extrusion and camera systems"
        ),
    },
}

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _maybe_int(s: str):
    try:
        return int(s)
    except (TypeError, ValueError):
        return s


def classify(path: Path, root: Path) -> dict | None:
    """Return a classification dict, or None if the path doesn't match.

    Expected layout: <root>/<dataset>/<Annotations|JPEGImages>/<split>/<vid>/<frame>.<ext>
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 5:
        return None
    dataset = parts[0]
    if dataset not in DATASETS:
        return None
    if "Annotations" in parts:
        image_type = "annotated"
    elif "JPEGImages" in parts:
        image_type = "raw"
    else:
        return None
    split = parts[2] if parts[2] in ("train", "test") else None
    vid_str = parts[3]
    frame_str = path.stem
    return {
        "dataset": dataset,
        "image_type": image_type,
        "split": split,
        "video_id": vid_str,
        "frame_id": frame_str,
        "video_number": _maybe_int(vid_str),
        "frame_number": _maybe_int(frame_str),
    }


def build_payload(c: dict) -> dict:
    ds = c["dataset"]
    return {
        "collection": COLLECTION,
        "authors": AUTHORS,
        "dataset": ds,
        "location": LOCATION,
        "image_type": c["image_type"],
        "split": c["split"],
        "video_number": c["video_number"],
        "frame_number": c["frame_number"],
        "classes": DATASETS[ds],
        "camera": CAMERAS[ds],
        "process": PROCESS[ds],
    }


def description_line(c: dict) -> str:
    ds = c["dataset"]
    cls = DATASETS[ds]
    cam = CAMERAS[ds]
    return (
        f"AMVOS · {ds} {c['image_type']} image; "
        f"video {c['video_id']} frame {c['frame_id']} ({c['split'] or 'unsplit'}); "
        f"{cam['make']} {cam['model']} @ {cam['fps']} fps ({cam['resolution']}, {cam['modality']}); "
        f"classes (255,255,255)={cls['(255,255,255)']}, (0,255,0)={cls['(0,255,0)']}; "
        f"collected at ORNL MDF, Oak Ridge, TN"
    )


def _image_unique_id(c: dict) -> str:
    """Stable 33-char-or-less ASCII ID for EXIF ImageUniqueID."""
    return f"{c['dataset']}-{c['split'] or 'na'}-V{c['video_id']}-F{c['frame_id']}"


def tag_jpeg(path: Path, c: dict) -> None:
    payload = build_payload(c)
    desc = description_line(c)
    cam = CAMERAS[c["dataset"]]
    try:
        exif_dict = piexif.load(str(path))
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    exif_dict.setdefault("0th", {})
    exif_dict.setdefault("Exif", {})

    exif_dict["0th"][piexif.ImageIFD.Artist] = AUTHORS_STR.encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.Copyright] = COPYRIGHT.encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = desc.encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.Software] = SOFTWARE.encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.Make] = cam["make"].encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.Model] = cam["model"].encode("utf-8")
    exif_dict["Exif"][piexif.ExifIFD.ImageUniqueID] = _image_unique_id(c).encode("ascii")
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = UserComment.dump(
        json.dumps(payload, separators=(",", ":")), encoding="unicode"
    )

    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, str(path))


# PNG text keys this script owns — never read back from existing chunks (would
# double-up on re-runs), always rewritten fresh.
_OWNED_PNG_KEYS = {
    "Author", "Copyright", "Description", "Software",
    "Collection", "Dataset", "Location", "ImageType", "Classes", "MetadataJSON",
    "Camera", "CameraMake", "CameraModel", "FrameRate", "Resolution",
    "Modality", "Mount", "VideoNumber", "FrameNumber", "Split",
    "Process", "Equipment",
}
_NON_TEXT_PNG_KEYS = {
    "dpi", "gamma", "transparency", "interlace", "srgb",
    "icc_profile", "chromaticity",
}


def tag_png(path: Path, c: dict) -> None:
    ds = c["dataset"]
    payload = build_payload(c)
    desc = description_line(c)
    cam = CAMERAS[ds]
    proc = PROCESS[ds]
    cls_json = json.dumps(DATASETS[ds], separators=(",", ":"))

    with Image.open(path) as img:
        mode = img.mode
        existing = dict(img.info)
        img.load()
        pixels = img.copy()

    info = PngImagePlugin.PngInfo()

    # Preserve any pre-existing textual chunks we don't own.
    skip = _OWNED_PNG_KEYS | _NON_TEXT_PNG_KEYS
    for k, v in existing.items():
        if k in skip:
            continue
        if isinstance(v, bytes):
            try:
                v = v.decode("utf-8")
            except UnicodeDecodeError:
                continue
        if isinstance(v, str):
            info.add_text(str(k), v)

    info.add_text("Author", AUTHORS_STR)
    info.add_text("Copyright", COPYRIGHT)
    info.add_text("Description", desc)
    info.add_text("Software", SOFTWARE)
    info.add_text("Collection", COLLECTION)
    info.add_text("Dataset", ds)
    info.add_text("Location", LOCATION)
    info.add_text("ImageType", c["image_type"])
    info.add_text("Classes", cls_json)
    info.add_text("CameraMake", cam["make"])
    info.add_text("CameraModel", cam["model"])
    info.add_text("Camera", f"{cam['make']} {cam['model']}")
    info.add_text("FrameRate", f"{cam['fps']} fps")
    info.add_text("Resolution", cam["resolution"])
    info.add_text("Modality", cam["modality"])
    info.add_text("Mount", cam["mount"])
    info.add_text("VideoNumber", str(c["video_id"]))
    info.add_text("FrameNumber", str(c["frame_id"]))
    info.add_text("Split", c["split"] or "")
    info.add_text("Process", proc["name"])
    info.add_text("Equipment", proc["equipment"])
    info.add_text("MetadataJSON", json.dumps(payload, separators=(",", ":")))

    save_kwargs = {"format": "PNG", "pnginfo": info, "optimize": False}
    if "icc_profile" in existing:
        save_kwargs["icc_profile"] = existing["icc_profile"]
    pixels.save(path, **save_kwargs)
    if mode != pixels.mode:
        # Defensive: PIL almost always preserves mode through copy+save,
        # but if it ever diverges we want a loud failure rather than silent loss.
        raise RuntimeError(f"mode changed: {mode} -> {pixels.mode} for {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--root", type=Path, required=True, help="Path to AMVOS/"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk and report only; do not modify files.",
    )
    ap.add_argument("--verbose", action="store_true", help="Log every file processed.")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"jpeg_tagged": 0, "png_tagged": 0, "errors": 0, "skipped_nonimage": 0}
    )

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in IMG_EXTS:
            continue
        c = classify(path, root)
        if c is None:
            stats["_unmatched"]["skipped_nonimage"] += 1
            if args.verbose:
                print(f"  skip (unmatched path): {path.relative_to(root)}")
            continue
        dataset, image_type = c["dataset"], c["image_type"]
        tag_str = (
            f"{dataset}/{image_type} V{c['video_id']}/F{c['frame_id']}"
            f"{' ' + c['split'] if c['split'] else ''}"
        )

        try:
            if args.dry_run:
                if args.verbose:
                    print(f"  would tag [{tag_str}] {path.relative_to(root)}")
            else:
                if ext == ".png":
                    tag_png(path, c)
                else:
                    tag_jpeg(path, c)
                if args.verbose:
                    print(f"  tagged   [{tag_str}] {path.relative_to(root)}")
            key = "png_tagged" if ext == ".png" else "jpeg_tagged"
            stats[dataset][key] += 1
        except Exception as e:
            stats[dataset]["errors"] += 1
            print(f"ERROR tagging {path}: {e!r}", file=sys.stderr)

    print()
    print(f"{'dataset':<14}{'jpeg':>8}{'png':>8}{'errors':>10}")
    print("-" * 40)
    total_j = total_p = total_e = 0
    for ds in sorted(k for k in stats if not k.startswith("_")):
        s = stats[ds]
        print(f"{ds:<14}{s['jpeg_tagged']:>8}{s['png_tagged']:>8}{s['errors']:>10}")
        total_j += s["jpeg_tagged"]
        total_p += s["png_tagged"]
        total_e += s["errors"]
    print("-" * 40)
    print(f"{'TOTAL':<14}{total_j:>8}{total_p:>8}{total_e:>10}")
    if stats["_unmatched"]["skipped_nonimage"]:
        print(
            f"\n(unmatched image-like files skipped: {stats['_unmatched']['skipped_nonimage']})"
        )
    print(f"\n{'DRY RUN -- no files modified' if args.dry_run else 'done.'}")
    return 1 if total_e else 0


if __name__ == "__main__":
    raise SystemExit(main())
