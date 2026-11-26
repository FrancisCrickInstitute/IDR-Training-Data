import argparse
import concurrent.futures
import csv
import glob
import json
import logging
import os
import random
import re
import shutil
import threading
import time

import imageio.v3 as iio
import numpy as np
from omero.gateway import BlitzGateway
from skimage.transform import resize
from tqdm import tqdm

MIN_SIZE = 256
MAX_CROP_SIZE = 1024

# Global set to track used image IDs across all threads
used_image_ids = set()
used_image_ids_lock = threading.Lock()

# One row per generated image pair, written by all worker threads
metadata_lock = threading.Lock()
METADATA_FIELDS = [
    'mixed_file', 'source_file', 'image_id', 'image_name', 'hierarchy',
    'project_or_screen_id', 'project_or_screen_name', 'dataset_or_plate_id', 'dataset_or_plate_name', 'well_id',
    'imaging_method', 'cell_line',
    'pixel_type', 'size_x', 'size_y', 'size_z', 'size_c', 'size_t',
    'z', 't', 'crop_x', 'crop_y', 'crop_size',
    'source_channel', 'target_channel', 'source_channel_name', 'target_channel_name',
    'crosstalk_coefficient', 'bleedthrough_proportion', 'intensity_scale',
    'image_annotations', 'study_annotations',
]


def prepopulate_used_image_ids(args):
    """
    Prepopulate the used_image_ids set by extracting image IDs from existing files
    in the output directories.
    """
    directories_to_check = []
    if args.mixed_dir and os.path.exists(args.mixed_dir):
        directories_to_check.append(args.mixed_dir)
    if args.source_dir and os.path.exists(args.source_dir):
        directories_to_check.append(args.source_dir)

    # Pattern to match filenames like: image_12345_alpha_0.25_mixed.tif
    pattern = re.compile(r'image_(\d+)_alpha_[\d.]+_(?:mixed|source|target)\.tif')

    extracted_ids = set()

    for directory in directories_to_check:
        print(f"Checking existing files in: {directory}")
        tif_files = glob.glob(os.path.join(directory, "*.tif"))

        for file_path in tif_files:
            filename = os.path.basename(file_path)
            match = pattern.match(filename)
            if match:
                image_id = int(match.group(1))
                extracted_ids.add(image_id)

    # Update the global used_image_ids set
    global used_image_ids
    with used_image_ids_lock:
        used_image_ids.update(extracted_ids)

    print(f"Prepopulated used_image_ids with {len(extracted_ids)} existing image IDs")
    logging.info(f"Prepopulated used_image_ids with {len(extracted_ids)} existing image IDs")

    return len(extracted_ids)


def clean_output_directories(args):
    """
    Check if output directories exist and clean them if they do.
    Create them if they don't exist.
    """
    directories = [args.mixed_dir, args.source_dir]

    for directory in directories:
        if directory is None:
            continue

        if os.path.exists(directory):
            print(f"Directory {directory} exists. Cleaning contents...")
            # Remove all files in the directory
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
        else:
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)


def print_obj(obj, indent=0):
    """
    Helper method to display info about OMERO objects.
    Now uses logging instead of printing to stdout.
    """
    logging.info("""%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        obj.OMERO_CLASS,
        obj.getId(),
        obj.getName(),
        obj.getOwnerOmeName()))


def is_image_already_used(image_id):
    """Check if an image ID has already been processed."""
    with used_image_ids_lock:
        return image_id in used_image_ids


def mark_image_as_used(image_id):
    """Mark an image ID as processed."""
    with used_image_ids_lock:
        used_image_ids.add(image_id)
        return True


def append_metadata_row(path, row):
    """Append one row to the metadata CSV, writing the header if the file is new."""
    with metadata_lock:
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def load_numpy_array(image, max_crop_size=MAX_CROP_SIZE):
    """
    Loads a random Z/T plane of every channel of a microscopy image, cropped to a
    random square region no larger than max_crop_size.

    Returns: (array of shape (C, crop_size, crop_size), dict describing the plane and crop),
             or (None, None) on failure.
    """
    pixels = image.getPrimaryPixels()
    size_z = pixels.getSizeZ()
    size_c = pixels.getSizeC()
    size_t = pixels.getSizeT()
    size_y = pixels.getSizeY()
    size_x = pixels.getSizeX()

    if size_z < 1 or size_t < 1 or size_c < 1:
        logging.warning(
            f"Image {image.getName()} has invalid dimensions (Z:{size_z}, C:{size_c}, T:{size_t}). Cannot load planes.")
        return None, None

    selected_z_plane = random.randint(0, size_z - 1)
    selected_t_timepoint = random.randint(0, size_t - 1)
    s = f"t:{size_t} c:{size_c} z:{size_z} y:{size_y} x:{size_x} (selected t:{selected_t_timepoint} z:{selected_z_plane})"
    logging.info(s)

    # Square crop so that resizing to MIN_SIZE x MIN_SIZE does not distort the image
    crop_size = min(max_crop_size, size_x, size_y)
    start_y = random.randint(0, size_y - crop_size)
    start_x = random.randint(0, size_x - crop_size)

    loaded_planes = []
    logging.info(
        f"Downloading image {image.getName()} (cropped region: X={start_x}-{start_x + crop_size}, Y={start_y}-{start_y + crop_size})")
    try:
        for c in range(size_c):
            tile_coords_tuple = (start_x, start_y, crop_size, crop_size)
            plane_data = pixels.getTile(selected_z_plane, c, selected_t_timepoint, tile_coords_tuple)
            loaded_planes.append(plane_data)
        all_channels_cropped_data = np.stack(loaded_planes)
    except Exception as e:
        logging.error(f"Error loading cropped planes for image {image.getName()}: {e}")
        return None, None

    crop_info = {
        'z': selected_z_plane,
        't': selected_t_timepoint,
        'crop_x': start_x,
        'crop_y': start_y,
        'crop_size': crop_size,
    }
    return all_channels_cropped_data, crop_info


def save_images(image: np.ndarray, filename: str, new_shape=None):
    if new_shape is not None:
        image = resize(image, new_shape, preserve_range=True)
    iio.imwrite(filename, image.astype(np.float32), extension='.tif')
    logging.info(f"Saved {filename}")


def generate_crosstalk_data(pure_target_channel: np.ndarray, pure_source_channel: np.ndarray,
                            crosstalk_coefficient: float) -> tuple[np.ndarray, float]:
    """
    Adds a scaled copy of the source channel to the target channel.

    The source is scaled so that roughly crosstalk_coefficient of the total intensity of the
    mixed image comes from the source. Returns the mixed image in the original (raw) intensity
    units and the measured bleed-through proportion, or (None, 0.0) if the source is empty.
    """
    if not (0.0 <= crosstalk_coefficient <= 1.0):
        logging.warning("Crosstalk coefficient is typically between 0 and 1.")
    if pure_target_channel.shape != pure_source_channel.shape:
        raise ValueError("Pure target and source channel images must have the same shape.")
    pure_target_channel = pure_target_channel.astype(np.float64)
    pure_source_channel = pure_source_channel.astype(np.float64)
    sum_pure_target_channel = float(np.sum(pure_target_channel))
    sum_pure_source_channel = float(np.sum(pure_source_channel))
    if not sum_pure_source_channel > 0.0:
        return None, 0.0
    alpha = (crosstalk_coefficient * sum_pure_target_channel) / (
            sum_pure_source_channel * (1.0 - crosstalk_coefficient))
    bleed_through_signal = alpha * pure_source_channel
    mixed_target_channel = pure_target_channel + bleed_through_signal
    sum_mixed_target_channel = float(np.sum(mixed_target_channel))
    if sum_mixed_target_channel > 0.0:
        bleedthrough_proportion = float(np.sum(bleed_through_signal)) / sum_mixed_target_channel
    else:
        bleedthrough_proportion = 0.0
    return mixed_target_channel, bleedthrough_proportion


HOST = 'ws://idr.openmicroscopy.org/omero-ws'
USER = 'public'
PASS = 'public'

MAX_RETRIES = 1000  # Increased to handle duplicate avoidance


def get_map_annotations(obj):
    """Returns all key-value (map) annotations on an OMERO object as {key: [values]}."""
    values = {}
    for annotation in obj.listAnnotations():
        if hasattr(annotation, 'getMapValue'):
            for key_value_pair in annotation.getMapValue():
                values.setdefault(key_value_pair.name, []).append(key_value_pair.value)
    return values


def has_imaging_method_annotation(obj, attribute_name="Imaging Method", search_terms=None):
    """
    Check if an object has the required imaging method annotation.

    Args:
        obj: OMERO object (Project or Screen)
        attribute_name: The annotation key to search for
        search_terms: List of terms to search for in the annotation value

    Returns:
        bool: True if the object has a matching annotation
    """
    if search_terms is None:
        search_terms = ["fluorescence", "confocal"]

    return any(term.lower() in value.lower()
               for value in get_map_annotations(obj).get(attribute_name, [])
               for term in search_terms)


def is_valid_image(image):
    """
    Check if an image meets the required criteria and hasn't been used.

    Args:
        image: OMERO image object

    Returns:
        bool: True if the image is valid and unused
    """
    return (image.getPrimaryPixels().getSizeC() > 1 and
            image.getPrimaryPixels().getSizeX() > MIN_SIZE and
            image.getPrimaryPixels().getSizeY() > MIN_SIZE and
            not is_image_already_used(image.getId()))


def find_random_project_image(conn):
    """
    Finds a random image from the Project > Dataset hierarchy that hasn't been used.
    Returns: A tuple of (project, dataset, image) or (None, None, None) on failure.
    """
    projects = list(conn.getObjects("Project"))

    retry_count = 0
    while retry_count < MAX_RETRIES:
        retry_count += 1
        random_project = random.choice(projects)

        if not has_imaging_method_annotation(random_project):
            continue

        datasets = list(random_project.listChildren())
        if not datasets:
            continue

        random_dataset = random.choice(datasets)
        images = list(random_dataset.listChildren())
        if not images:
            continue

        # Try multiple images from this dataset to find an unused one
        random.shuffle(images)
        for image in images[:min(10, len(images))]:
            if is_valid_image(image):
                mark_image_as_used(image.getId())
                logging.info(f"Successfully found unused project image after {retry_count} attempts.")
                return random_project, random_dataset, image

    logging.error(f"Failed to find an unused Project image after {MAX_RETRIES} attempts.")
    return None, None, None


def find_random_screen_image(conn):
    """
    Finds a random image from the Screen > Plate > Well hierarchy that hasn't been used.
    Returns: A tuple of (screen, plate, well, image) or (None, None, None, None) on failure.
    """
    screens = list(conn.getObjects("Screen"))
    if not screens:
        logging.error("No screens found on the server.")
        return None, None, None, None

    retry_count = 0
    while retry_count < MAX_RETRIES:
        retry_count += 1
        random_screen = random.choice(screens)

        if not has_imaging_method_annotation(random_screen):
            continue

        plates = list(random_screen.listChildren())
        if not plates:
            continue

        random_plate = random.choice(plates)
        wells = list(random_plate.listChildren())
        if not wells:
            continue

        random_well = random.choice(wells)
        well_samples = list(random_well.listChildren())
        if not well_samples:
            continue

        # Try multiple images from this well to find an unused one
        random.shuffle(well_samples)
        for well_sample in well_samples[:min(5, len(well_samples))]:
            image = well_sample.getImage()
            if is_valid_image(image):
                mark_image_as_used(image.getId())
                logging.info(f"Successfully found unused screen image after {retry_count} attempts.")
                return random_screen, random_plate, random_well, image

    logging.error(f"Failed to find an unused Screen image after {MAX_RETRIES} attempts.")
    return None, None, None, None


def pick_random_image(conn, worker_id):
    """
    Picks an unused image from either the Project or the Screen hierarchy at random.
    Returns: (image, context dict describing where it came from), or (None, None).
    """
    if random.choice([0, 1]) == 0:
        logging.info(f"Worker {worker_id}: Choosing image from Project hierarchy.")
        study, container, image = find_random_project_image(conn)
        if image is None:
            return None, None
        for obj in (study, container, image):
            print_obj(obj)
        well_id = ''
        hierarchy = 'Project/Dataset'
    else:
        logging.info(f"Worker {worker_id}: Choosing image from Screen hierarchy.")
        study, container, well, image = find_random_screen_image(conn)
        if image is None:
            return None, None
        for obj in (study, container, well, image):
            print_obj(obj)
        well_id = well.getId()
        hierarchy = 'Screen/Plate/Well'

    study_annotations = get_map_annotations(study)
    image_annotations = get_map_annotations(image)
    context = {
        'hierarchy': hierarchy,
        'project_or_screen_id': study.getId(),
        'project_or_screen_name': study.getName(),
        'dataset_or_plate_id': container.getId(),
        'dataset_or_plate_name': container.getName(),
        'well_id': well_id,
        'imaging_method': '; '.join(study_annotations.get('Imaging Method', [])),
        'cell_line': '; '.join(image_annotations.get('Cell Line', [])),
        'image_annotations': json.dumps(image_annotations),
        'study_annotations': json.dumps(study_annotations),
    }
    return image, context


def process_image(args, image, context):
    """Generates, saves and records one mixed/source image pair. Returns True on success."""
    data, crop_info = load_numpy_array(image)
    if data is None:
        logging.error("No image has been loaded - something has gone wrong somewhere!")
        return False

    source_channel, target_channel = random.sample(range(data.shape[0]), 2)
    logging.info(f'Source Channel: {source_channel}')
    logging.info(f'Target Channel: {target_channel}')

    crosstalk_coefficient = random.random() * args.max_crosstalk
    mixed_image, crosstalk_proportion = generate_crosstalk_data(
        pure_target_channel=data[target_channel],
        pure_source_channel=data[source_channel],
        crosstalk_coefficient=crosstalk_coefficient
    )
    if mixed_image is None:
        logging.warning(f"Source channel {source_channel} of image {image.getId()} is empty - skipping.")
        return False

    # Put both images on the same intensity scale so the pair can be compared directly
    source_image = data[source_channel].astype(np.float64)
    intensity_scale = float(max(np.max(mixed_image), np.max(source_image)))

    mixed_filename = os.path.join(args.mixed_dir,
                                  f"image_{image.getId()}_alpha_{crosstalk_proportion:.2f}_mixed.tif")
    source_filename = os.path.join(args.source_dir,
                                   f"image_{image.getId()}_alpha_{crosstalk_proportion:.2f}_source.tif")
    save_images(mixed_image / intensity_scale, mixed_filename, new_shape=[MIN_SIZE, MIN_SIZE])
    save_images(source_image / intensity_scale, source_filename, new_shape=[MIN_SIZE, MIN_SIZE])

    pixels = image.getPrimaryPixels()
    channel_labels = image.getChannelLabels() or []
    append_metadata_row(args.metadata_csv, {
        **context,
        **crop_info,
        'mixed_file': os.path.basename(mixed_filename),
        'source_file': os.path.basename(source_filename),
        'image_id': image.getId(),
        'image_name': image.getName(),
        'pixel_type': image.getPixelsType(),
        'size_x': pixels.getSizeX(),
        'size_y': pixels.getSizeY(),
        'size_z': pixels.getSizeZ(),
        'size_c': pixels.getSizeC(),
        'size_t': pixels.getSizeT(),
        'source_channel': source_channel,
        'target_channel': target_channel,
        'source_channel_name': channel_labels[source_channel] if source_channel < len(channel_labels) else '',
        'target_channel_name': channel_labels[target_channel] if target_channel < len(channel_labels) else '',
        'crosstalk_coefficient': crosstalk_coefficient,
        'bleedthrough_proportion': crosstalk_proportion,
        'intensity_scale': intensity_scale,
    })
    return True


def worker_task(args, images_to_process, worker_id):
    if images_to_process == 0:
        return

    logging.info(
        f"Worker {worker_id} starting. Assigned to process {images_to_process} images. Creating new connection...")
    conn = None
    try:
        conn = BlitzGateway(USER, PASS, host=HOST, secure=True)
        conn.connect()
        conn.c.enableKeepAlive(60)

        successful_processes = 0
        for i in range(images_to_process):
            logging.info(f'Worker {worker_id}: Processing image {i + 1} of {images_to_process}.')

            image, context = pick_random_image(conn, worker_id)
            if image is None:
                logging.warning(
                    f"Worker {worker_id}: Could not find a suitable unused image to process in this iteration.")
                continue

            if process_image(args, image, context):
                successful_processes += 1

        logging.info(f"Worker {worker_id} completed. Successfully processed {successful_processes} images.")

    except Exception as e:
        logging.error(f"An error occurred in worker {worker_id}: {e}")
    finally:
        if conn and conn.isConnected():
            conn.close()
            logging.info(f"Worker {worker_id} finished. Connection closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--max_crosstalk', type=float, default=0.5,
                        help='Maximum relative crosstalk applied from source channel to bleed channel')
    parser.add_argument('-b', '--mixed_dir', required=True, help='Directory to save generated "bleed-through" images')
    parser.add_argument('-s', '--source_dir', required=True, help='Directory to save original "source" channel')
    parser.add_argument('-n', '--number_of_images', type=int, default=1,
                        help='Number of image sets to generate')
    parser.add_argument('-l', '--log_file', type=str, default='processing.log',
                        help='Path to the log file for detailed output')
    parser.add_argument('-m', '--metadata_csv', type=str, default=None,
                        help='CSV file to append per-image metadata to (default: metadata.csv in the mixed directory)')
    parser.add_argument('-p', '--skip-prepopulate', action='store_true',
                        help='Skip prepopulating used image IDs from existing files')
    args = parser.parse_args()
    if args.metadata_csv is None:
        args.metadata_csv = os.path.join(args.mixed_dir, 'metadata.csv')

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(args.log_file, mode='w')])

    print(f"Detailed logs are being written to {args.log_file}")
    print(f"Per-image metadata is being appended to {args.metadata_csv}")

    # Clean/create output directories (MODIFIED: only clean if not prepopulating)
    if args.skip_prepopulate:
        clean_output_directories(args)
    else:
        # Create directories if they don't exist, but don't clean them
        directories = [args.mixed_dir, args.source_dir]
        for directory in directories:
            if directory and not os.path.exists(directory):
                print(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)

        # Prepopulate used image IDs from existing files
        prepopulate_used_image_ids(args)

    start_time = time.time()

    num_workers = min(args.number_of_images, os.cpu_count() * 4)
    images_per_worker = args.number_of_images // num_workers
    remainder = args.number_of_images % num_workers
    work_items = [images_per_worker] * num_workers
    for i in range(remainder):
        work_items[i] += 1

    print(f"Total images to generate: {args.number_of_images}")
    print(f"Number of workers: {num_workers}")
    print(f"Work distribution: {work_items}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_task, args, count, i) for i, count in enumerate(work_items)]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            try:
                future.result()
            except Exception as e:
                logging.exception("An exception from a thread was propagated.")

    end_time = time.time()
    total_time = end_time - start_time

    print("\nAll tasks completed. Please check the log file for details.")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Total unique images processed: {len(used_image_ids)}")
