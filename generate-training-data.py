import argparse
import concurrent.futures
import logging
import os
import random
import threading
import time

import imageio.v3 as iio
import numpy as np
from omero.gateway import BlitzGateway
from skimage.transform import resize
from tqdm import tqdm

MIN_SIZE = 256
NORM_COEFF = np.power(2, 16)

# Global set to track used image IDs across all threads
used_image_ids = set()
used_image_ids_lock = threading.Lock()


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


def load_numpy_array(image, target_x_dim=1024, target_y_dim=1024):
    """
    Loads a microscopy image into a NumPy array.
    Uses logging for status messages.
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
        return None

    selected_z_plane = random.randint(0, size_z - 1)
    selected_t_timepoint = random.randint(0, size_t - 1)
    s = f"t:{size_t} c:{size_c} z:{size_z} y:{size_y} x:{size_x} (selected t:{selected_t_timepoint} z:{selected_z_plane})"
    logging.info(s)

    cropped_size_y = min(target_y_dim, size_y)
    cropped_size_x = min(target_x_dim, size_x)
    start_y = random.randint(0, size_y - cropped_size_y) if size_y > target_y_dim else 0
    start_x = random.randint(0, size_x - cropped_size_x) if size_x > target_x_dim else 0

    loaded_planes = []
    logging.info(
        f"Downloading image {image.getName()} (cropped region: X={start_x}-{start_x + cropped_size_x}, Y={start_y}-{start_y + cropped_size_y})")
    try:
        for c in range(size_c):
            tile_coords_tuple = (start_x, start_y, cropped_size_x, cropped_size_y)
            plane_data = pixels.getTile(selected_z_plane, c, selected_t_timepoint, tile_coords_tuple)
            loaded_planes.append(plane_data)
        all_channels_cropped_data = np.stack(loaded_planes)
    except Exception as e:
        logging.error(f"Error loading cropped planes for image {image.getName()}: {e}")
        return None

    final_shape = (1, size_c, 1, cropped_size_y, cropped_size_x)
    return np.reshape(all_channels_cropped_data, newshape=final_shape)


def save_images(image: np.ndarray, filename: str, new_shape=None):
    if new_shape is not None:
        image = resize(image, new_shape, preserve_range=True)
    iio.imwrite(filename, image, extension='.tif')
    logging.info(f"Saved {filename}")


def generate_crosstalk_data(pure_target_channel: np.ndarray, pure_source_channel: np.ndarray,
                            crosstalk_coefficient: float) -> tuple[np.ndarray, float]:
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
    current_max = np.max(mixed_target_channel)
    if current_max == 0:
        normalized_mixed_target_channel = np.zeros_like(mixed_target_channel)
    else:
        normalized_mixed_target_channel = mixed_target_channel * NORM_COEFF / current_max
    sum_mixed_target_channel = float(np.sum(mixed_target_channel))
    if sum_mixed_target_channel > 0.0:
        bleedthrough_proportion = float(np.sum(bleed_through_signal)) / sum_mixed_target_channel
    else:
        bleedthrough_proportion = 0.0
    return normalized_mixed_target_channel / NORM_COEFF, bleedthrough_proportion


HOST = 'ws://idr.openmicroscopy.org/omero-ws'
USER = 'public'
PASS = 'public'

MAX_RETRIES = 1000  # Increased to handle duplicate avoidance


def find_random_project_image(conn):
    """
    Finds a random image from the Project > Dataset hierarchy that hasn't been used.
    Returns: A tuple of (project, dataset, image) or (None, None, None) on failure.
    """
    projects = list(conn.getObjects("Project"))
    attribute_name = "Imaging Method"
    search_terms = ["fluorescence", "confocal"]

    retry_count = 0
    while retry_count < MAX_RETRIES:
        retry_count += 1
        random_project = random.choice(projects)
        kv_annotations = random_project.listAnnotations()
        found_imaging_method = False
        for annotation in kv_annotations:
            if hasattr(annotation, 'getMapValue'):
                for key_value_pair in annotation.getMapValue():
                    key = key_value_pair.name
                    value = key_value_pair.value
                    if key == attribute_name:
                        if any(term.lower() in value.lower() for term in search_terms):
                            found_imaging_method = True
                            break
            if found_imaging_method:
                break

        if not found_imaging_method:
            continue

        datasets = list(random_project.listChildren())
        if not datasets:
            continue

        random_dataset = random.choice(datasets)
        images = list(random_dataset.listChildren())
        if not images:
            continue

        # Try multiple images from this dataset to find an unused one
        random.shuffle(images)  # Randomize the order
        for image in images[:min(10, len(images))]:  # Check up to 10 images from this dataset
            if (image.getPrimaryPixels().getSizeC() > 1 and
                    image.getPrimaryPixels().getSizeX() > MIN_SIZE and
                    image.getPrimaryPixels().getSizeY() > MIN_SIZE and
                    not is_image_already_used(image.getId())):
                # Mark as used immediately to prevent race conditions
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
        random.shuffle(well_samples)  # Randomize the order
        for well_sample in well_samples[:min(5, len(well_samples))]:  # Check up to 5 images from this well
            image = well_sample.getImage()
            if (image.getPrimaryPixels().getSizeC() > 1 and
                    image.getPrimaryPixels().getSizeX() > MIN_SIZE and
                    image.getPrimaryPixels().getSizeY() > MIN_SIZE and
                    not is_image_already_used(image.getId())):
                # Mark as used immediately to prevent race conditions
                mark_image_as_used(image.getId())
                logging.info(f"Successfully found unused screen image after {retry_count} attempts.")
                return random_screen, random_plate, random_well, image

    logging.error(f"Failed to find an unused Screen image after {MAX_RETRIES} attempts.")
    return None, None, None, None


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

            image = None
            # Randomly choose between Project hierarchy (0) and Screen hierarchy (1)
            hierarchy_choice = random.choice([0, 1])

            if hierarchy_choice == 0:
                logging.info(f"Worker {worker_id}: Choosing image from Project hierarchy.")
                random_project, random_dataset, random_image = find_random_project_image(conn)
                if random_image:
                    print_obj(random_project)
                    print_obj(random_dataset)
                    print_obj(random_image)
                    image = random_image
            else:
                logging.info(f"Worker {worker_id}: Choosing image from Screen hierarchy.")
                random_screen, random_plate, random_well, random_image = find_random_screen_image(conn)
                if random_image:
                    print_obj(random_screen)
                    print_obj(random_plate)
                    print_obj(random_well)
                    print_obj(random_image)
                    image = random_image

            if image is not None:
                data = load_numpy_array(image)
                if data is not None:
                    source_channel = random.choice(range(np.size(data[0], 0)))
                    target_channel = source_channel
                    while source_channel == target_channel:
                        target_channel = random.choice(range(np.size(data[0], 0)))

                    logging.info(f'Source Channel: {source_channel}')
                    logging.info(f'Target Channel: {target_channel}')

                    crosstalk_coefficients = [random.random() * args.max_crosstalk]
                    for alpha in crosstalk_coefficients:
                        mixed_image, crosstalk_proportion = generate_crosstalk_data(
                            pure_target_channel=data[0, target_channel, 0],
                            pure_source_channel=data[0, source_channel, 0],
                            crosstalk_coefficient=alpha
                        )
                        if mixed_image is not None:
                            mixed_filename = os.path.join(args.mixed_dir,
                                                          f"image_{image.getId()}_alpha_{crosstalk_proportion:.2f}_mixed.tif")
                            source_filename = os.path.join(args.source_dir,
                                                           f"image_{image.getId()}_alpha_{crosstalk_proportion:.2f}_source.tif")
                            target_filename = os.path.join(args.target_dir,
                                                           f"image_{image.getId()}_alpha_{crosstalk_proportion:.2f}_target.tif")
                            save_images(mixed_image, mixed_filename, new_shape=[MIN_SIZE, MIN_SIZE])
                            logging.info(f'Generated and saved {mixed_filename} in {args.mixed_dir}')
                            save_images(data[0, source_channel, 0] / NORM_COEFF, source_filename,
                                        new_shape=[MIN_SIZE, MIN_SIZE])
                            save_images(data[0, target_channel, 0] / NORM_COEFF, target_filename,
                                        new_shape=[MIN_SIZE, MIN_SIZE])
                            logging.info(f'Generated and saved {source_filename} in {args.source_dir}')
                            successful_processes += 1
                else:
                    logging.error("No image has been loaded - something has gone wrong somewhere!")
            else:
                logging.warning(
                    f"Worker {worker_id}: Could not find a suitable unused image to process in this iteration.")

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
    parser.add_argument('-b', '--mixed_dir', help='Directory to save generated "bleed-through" images')
    parser.add_argument('-s', '--source_dir', help='Directory to save original "source" channel')
    parser.add_argument('-t', '--target_dir', help='Directory to save original "target" channel')
    parser.add_argument('-n', '--number_of_images', type=int, default=1,
                        help='Number of image sets to generate')
    parser.add_argument('-l', '--log_file', type=str, default='processing.log',
                        help='Path to the log file for detailed output')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(args.log_file, mode='w')])

    print(f"Detailed logs are being written to {args.log_file}")
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
