# Datasets

Converts Roboflow COCO annotations into VOC-style segmentation masks and builds the original SAM 2 dataset layout for the additive manufacturing datasets. It also embeds authorship and capture metadata into the released **AMVOS** (Additive Manufacturing Video Object Segmentation) dataset. Part of the ORNLxUTK [DomainSpecific](https://github.com/ORNLxUTK/DomainSpecific) project.

AMVOS is publicly available on [Harvard Dataverse](https://doi.org/10.7910/DVN/5GSQTS) and described in the [Data in Brief article](https://doi.org/10.1016/j.dib.2026.113249).

## Datasets

| Dataset | Process | Labeled classes (white / green) |
|---------|---------|---------------------------------|
| TIG | Tungsten inert gas wire arc additive manufacturing (TIG-WAAM) | Feed Wire / Melt Pool |
| PAW (PLASMA) | Plasma arc welding | Feed Wire / Melt Pool |
| LHW-DED (MAZAK) | Laser hot-wire directed energy deposition | Feed Wire / Melt Pool |
| visPOLYMER | Polymer pellet extrusion, visible imaging | Nozzle / Material |
| irPOLYMER | Polymer pellet extrusion, infrared imaging | Nozzle / Material |

## Setup

```bash
uv sync
```

## Scripts

| Script | Purpose |
|--------|---------|
| `roboflow_to_annotationimage.py` | Convert Roboflow `_annotations.coco.json` polygons into 3-channel PNG masks. White (255, 255, 255) = wire/nozzle; green (0, 255, 0) = melt pool/material |
| `createoriginaldirectorystructure.py` | Pair the original frames with their masks and build `SAM2images/<dataset>/{JPEGImages,Annotations}`, renaming frames to `00000.jpg`, `00001.jpg`, ... |
| `tag_dataset_metadata.py` | Embed authors, location, process, camera, and class metadata into every AMVOS image (EXIF for JPEG, text chunks for PNG) |

## Usage

```bash
# 1. COCO polygons → PNG masks (one subdirectory per dataset, each with train/_annotations.coco.json)
python roboflow_to_annotationimage.py roboflowdata/

# 2. Build the SAM 2 directory layout
#    Source frame directories are set in main()
python createoriginaldirectorystructure.py

# 3. Tag AMVOS images with metadata (modifies files in place)
python tag_dataset_metadata.py --root AMVOS --dry-run
python tag_dataset_metadata.py --root AMVOS
```

## Output Layout

```
SAM2images/<dataset>/
├── JPEGImages/{train,test}/<video_id>/00000.jpg, ...
└── Annotations/{train,test}/<video_id>/00000.png, ...
```

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{wetzel2026domainspecific,
  title={Domain-specific adaptation: low-rank adaptation fine-tuning of {SAM} 2 for manufacturing processes},
  author={Wetzel, C. and Haley, J. and Paquit, V. and Orlyanchik, V. and Santos-Villalobos, H.},
  journal={Journal of Intelligent Manufacturing},
  year={2026},
  doi={10.1007/s10845-026-02973-6}
}

@article{wetzel2026amvosdib,
  title={Cross domain additive manufacturing video object segmentation dataset},
  author={Wetzel, Calvin and Santos-Villalobos, Hector and Haley, James and Orlyanchik, Vladimir and Rodriguez Parra, Mario and Paramanathan, Mithulan and Feldhausen, Tom and Sebok, Michael and Masuo, Chris and Paquit, Vincent},
  journal={Data in Brief},
  pages={113249},
  year={2026},
  doi={10.1016/j.dib.2026.113249}
}

@misc{wetzel2026amvosdataset,
  title={{AMVOS}: Additive Manufacturing Video Object Segmentation Dataset},
  author={Wetzel, Calvin and Santos-Villalobos, Hector and Haley, James and Orlyanchik, Vladimir and Rodriguez Parra, Mario Alberto and Paramanathan, Mithulan and Feldhausen, Thomas and Sebok, Michael and Masuo, Christopher and Paquit, Vincent},
  publisher={Harvard Dataverse},
  version={V1},
  year={2026},
  doi={10.7910/DVN/5GSQTS}
}
```

## Article

[1] C. Wetzel, J. Haley, V. Paquit, V. Orlyanchik, H. Santos-Villalobos, Domain-specific adaptation: low-rank adaptation fine-tuning of SAM 2 for manufacturing processes, Journal of Intelligent Manufacturing (2026). https://doi.org/10.1007/s10845-026-02973-6

[2] C. Wetzel, H. Santos-Villalobos, J. Haley, V. Orlyanchik, M. Rodriguez Parra, M. Paramanathan, T. Feldhausen, M. Sebok, C. Masuo, V. Paquit, Cross domain additive manufacturing video object segmentation dataset, Data in Brief (2026) 113249. https://doi.org/10.1016/j.dib.2026.113249

[3] C. Wetzel, H. Santos-Villalobos, J. Haley, V. Orlyanchik, M.A. Rodriguez Parra, M. Paramanathan, T. Feldhausen, M. Sebok, C. Masuo, V. Paquit, AMVOS: Additive Manufacturing Video Object Segmentation Dataset, Harvard Dataverse, V1 (2026). https://doi.org/10.7910/DVN/5GSQTS
