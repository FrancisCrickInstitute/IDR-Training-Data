# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Generates training data for detecting fluorescence channel cross-talk ("bleed-through"). Scripts pull random multi-channel fluorescence/confocal images from the public Image Data Resource (IDR, `idr.openmicroscopy.org`), pick a random source and target channel, add a random amount of the source channel into the target channel, and save the mixed target plus the original source channel as 256×256 TIFFs.

## Working agreement

Read `WORKING_AGREEMENT.md` before starting. It sets out what you may do without asking, what needs Dave's "yes" first, and what you must never do. `.claude/settings.json` enforces part of it.

In particular, ask before you:
- edit `pixi.toml` or run any `pixi install`, `add` or `lock`
- move or delete files, including `git rm`
- change git state
- contact external servers, including IDR
- start long runs of the generator

## Environment

The project uses pixi (`pixi.toml`). Use `pixi run --no-install -e idr-training …`. `omero` is installed only in the `idr-training` environment, and a plain `pixi run` uses `default`, which doesn't have it. Without `--no-install`, `pixi run` may install packages, and installs are Dave's job. There is no `python` on PATH. omero's generated files print many `SyntaxWarning`s on import; they're harmless, and `python -W ignore::SyntaxWarning` silences them.

`omero-py` must come from conda-forge (`[dependencies]` in `pixi.toml`), **not** PyPI. Its dependency `zeroc-ice==3.6.5` is source-only on PyPI and fails to compile with modern MSVC (`'binary_function': is not a member of 'std'`). conda-forge has prebuilt Windows builds for current Python versions, so you don't need an older Python. `environment.yml` and `requirements.txt` are older conda/pip setups.

Syntax check without importing: `pixi run --no-install python -m py_compile <file>.py`.

Raw pixels have to come through OMERO (BlitzGateway with `getTile`). IDR's public HTTP endpoints serve only display-rendered 8-bit images, and its pixel-tile and zarr paths are not available, so don't build an HTTP-only downloader.

The repo has no test suite, linter config or build step.

## Running

Main generator (OMERO BlitzGateway, multithreaded):

```
pixi run --no-install -e idr-training python generate-training-data.py -b <mixed_dir> -s <source_dir> -n <num_images> [-c 0.5] [-l processing.log] [-m metadata.csv] [-p]
```

- One CSV row per generated pair is appended to `-m` (default `<mixed_dir>/metadata.csv`). It records the source study, container and well, imaging method, cell line, dimensions and pixel type, Z/T, the crop, channel indices and names, the input coefficient, the measured proportion, the intensity scale, and all image- and study-level map annotations as JSON.

- Detailed progress goes to the log file (`-l`, default `processing.log`, overwritten each run). Stdout only shows a tqdm bar and a summary.
- By default it keeps existing output and resumes: `prepopulate_used_image_ids` reads image IDs from existing filenames so those images are not used again.
- `-p/--skip-prepopulate` **deletes everything in the output directories** before starting.

Other scripts:
- `generate_idr_image_list.py -o image_details.csv -n N`: writes a CSV of randomly chosen images and the crop, Z/T and channel choices, without downloading pixels.
- `inspect_results.py`: reports mixed/source files that have no matching pair. The directory paths are hard-coded to a local `Z:/` share, so edit them before running.
- `GetIDRImages.ipynb`: notebook walkthrough of the same pipeline (Binder link in README).

## Architecture notes

- **Image selection** (`generate-training-data.py`): each worker picks either the Project→Dataset→Image or the Screen→Plate→Well→WellSample→Image hierarchy at random. A Project or Screen qualifies only if its map annotation `"Imaging Method"` contains "fluorescence" or "confocal" (`has_imaging_method_annotation`). An image qualifies only if it has more than one channel and X and Y larger than `MIN_SIZE` (256) (`is_valid_image`).
- **Retries**: some IDR images fail to load on the server side (`ResourceError: Error instantiating pixel buffer`). A worker keeps picking new images until it has saved its quota, trying at most `MAX_ATTEMPTS_PER_IMAGE` × quota images. The run ends by printing pairs saved against pairs requested.
- **Concurrency**: the script uses a `ThreadPoolExecutor` with up to `cpu_count*4` workers. Each worker opens its own `BlitzGateway` connection (public/public over `ws://…/omero-ws`). A global `used_image_ids` set, guarded by a lock, keeps images unique across threads.
- **Loading**: `load_numpy_array` picks a random Z and T and a random **square** crop with side `min(1024, X, Y)`, so the resize to 256×256 doesn't distort the image. It fetches each channel with `pixels.getTile` and returns `(array of shape (C, S, S), crop_info)`.
- **Cross-talk model**: in `generate-training-data.py`, `alpha` is scaled so that `crosstalk_coefficient` is the intended fraction of the summed mixed-image intensity that comes from bleed-through. `generate_crosstalk_data` returns the mixed image in raw units. `process_image` then divides the mixed and source images by the **same** scale (the larger of their two maxima, recorded as `intensity_scale` in the CSV) and saves both as float32. The notebook uses the simpler `coefficient * source` with the old per-image normalisation.
- **Filename convention**: the files are named `image_<omeroImageId>_alpha_<bleedProportion:.2f>_{mixed|source}.tif`, where the number is the measured bleed-through proportion, not the input coefficient. Resume logic (regex in `prepopulate_used_image_ids`) and `inspect_results.py` both parse this pattern, so keep them in step with any change to it.
- **Channel metadata is unreliable**: OMERO channel names are often placeholders (`0`, `1`, …), and transmitted-light channels have slipped into past training data. The image-level `Channels` map annotation is often better but doesn't always map onto channel numbers. Nothing filters transmitted light yet; see `docs/plans/transmitted-light-filter.md`.
- The scripts are standalone and share no module. Helpers such as `print_obj`, `save_images` and `generate_crosstalk_data` are duplicated across files and the notebook, so a fix in one place may need copying to the others.
