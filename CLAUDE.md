# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Generates training data for detecting fluorescence channel cross-talk ("bleed-through"). Scripts pull random multi-channel fluorescence/confocal images from the public Image Data Resource (IDR, `idr.openmicroscopy.org`), pick a random source and target channel, add a random amount of the source channel into the target channel, and save the mixed target plus the original source channel as 256×256 TIFFs.

## Environment

OMERO's Python bindings (`omero-py`) need `zeroc-ice==3.6.5`, which only installs cleanly through conda, so use the conda environment:

```
conda env create -f environment.yml   # creates env "omero" (Python 3.9)
conda activate omero
pip install -r requirements.txt       # tqdm, scikit-image, requests etc. are not in environment.yml
```

Raw pixels have to come through OMERO (BlitzGateway with `getTile`). IDR's public HTTP endpoints serve only display-rendered 8-bit images, and its pixel-tile and zarr paths are not available, so don't build an HTTP-only downloader.

The repo has no test suite, linter config or build step.

## Running

Main generator (OMERO BlitzGateway, multithreaded):

```
python generate-training-data.py -b <mixed_dir> -s <source_dir> -n <num_images> [-c 0.5] [-l processing.log] [-p]
```

- Detailed progress goes to the log file (`-l`, default `processing.log`, overwritten each run). Stdout only shows a tqdm bar and a summary.
- By default it keeps existing output and resumes: `prepopulate_used_image_ids` reads image IDs from existing filenames so those images are not used again.
- `-p/--skip-prepopulate` **deletes everything in the output directories** before starting.

Other scripts:
- `generate_idr_image_list.py -o image_details.csv -n N`: writes a CSV of randomly chosen images and the crop, Z/T and channel choices, without downloading pixels.
- `inspect_results.py`: reports mixed/source files that have no matching pair. The directory paths are hard-coded to a local `Z:/` share, so edit them before running.
- `GetIDRImages.ipynb`: notebook walkthrough of the same pipeline (Binder link in README).

## Architecture notes

- **Image selection** (`generate-training-data.py`): each worker picks either the Project→Dataset→Image or the Screen→Plate→Well→WellSample→Image hierarchy at random. A Project or Screen qualifies only if its map annotation `"Imaging Method"` contains "fluorescence" or "confocal" (`has_imaging_method_annotation`). An image qualifies only if it has more than one channel and X and Y larger than `MIN_SIZE` (256) (`is_valid_image`).
- **Concurrency**: the script uses a `ThreadPoolExecutor` with up to `cpu_count*4` workers. Each worker opens its own `BlitzGateway` connection (public/public over `ws://…/omero-ws`). A global `used_image_ids` set, guarded by a lock, keeps images unique across threads.
- **Loading**: `load_numpy_array` picks a random Z and T and a random crop of at most 1024×1024, then fetches each channel with `pixels.getTile`. It returns an array of shape `(1, C, 1, Y, X)`.
- **Cross-talk model**: in `generate-training-data.py`, `alpha` is scaled so that `crosstalk_coefficient` is the intended fraction of the summed mixed-image intensity that comes from bleed-through. The notebook uses the simpler `coefficient * source`. The mixed image is normalised to its max (range 0–1). The source image is divided by 2^16.
- **Filename convention**: the files are named `image_<omeroImageId>_alpha_<bleedProportion:.2f>_{mixed|source}.tif`, where the number is the measured bleed-through proportion, not the input coefficient. Resume logic (regex in `prepopulate_used_image_ids`) and `inspect_results.py` both parse this pattern, so keep them in step with any change to it.
- The scripts are standalone and share no module. Helpers such as `print_obj`, `save_images` and `generate_crosstalk_data` are duplicated across files and the notebook, so a fix in one place may need copying to the others.
