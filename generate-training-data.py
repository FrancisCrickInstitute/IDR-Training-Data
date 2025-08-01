import argparse
import concurrent.futures
import logging
import os
import random
import time

import imageio.v3 as iio
import numpy as np
from omero.gateway import BlitzGateway
from skimage.transform import resize
from tqdm import tqdm

MIN_SIZE = 256
NORM_COEFF = np.power(2, 16)


# All print_obj statements will now go to the log file
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


def worker_task(args, images_to_process):
    if images_to_process == 0:
        return

    logging.info(f"Thread starting. Assigned to process {images_to_process} images. Creating new connection...")
    conn = None
    try:
        conn = BlitzGateway(USER, PASS, host=HOST, secure=True)
        conn.connect()
        conn.c.enableKeepAlive(60)

        for i in range(images_to_process):
            logging.info(f'Processing image {i + 1} of {images_to_process} in this thread.')

            projects = list(conn.getObjects("Project"))
            attribute_name = "Imaging Method"
            search_terms = ["fluorescence", "confocal"]

            foundProject = False
            while not foundProject:
                random_project = random.choice(projects)
                kv_annotations = random_project.listAnnotations()
                for annotation in kv_annotations:
                    if hasattr(annotation, 'getMapValue'):
                        for key_value_pair in annotation.getMapValue():
                            key = key_value_pair.name
                            value = key_value_pair.value
                            if key == attribute_name:
                                if any(term.lower() in value.lower() for term in search_terms):
                                    datasets = list(random_project.listChildren())
                                    random_dataset = random.choice(datasets)
                                    images = list(random_dataset.listChildren())
                                    random_image = random.choice(images)
                                    if (random_image.getPrimaryPixels().getSizeC() > 1 and
                                            random_image.getPrimaryPixels().getSizeX() > MIN_SIZE and
                                            random_image.getPrimaryPixels().getSizeY() > MIN_SIZE):
                                        foundProject = True
                                        break

            # Note: print_obj now uses the logger
            print_obj(random_project)
            print_obj(random_dataset)
            print_obj(random_image)

            data = load_numpy_array(random_image)
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
                                                      f"image_{random_image.getId()}_alpha_{crosstalk_proportion:.2f}_mixed.tif")
                        source_filename = os.path.join(args.source_dir,
                                                       f"image_{random_image.getId()}_alpha_{crosstalk_proportion:.2f}_source.tif")
                        save_images(mixed_image, mixed_filename, new_shape=[MIN_SIZE, MIN_SIZE])
                        logging.info(f'Generated and saved {mixed_filename} in {args.mixed_dir}')
                        save_images(data[0, source_channel, 0] / NORM_COEFF, source_filename,
                                    new_shape=[MIN_SIZE, MIN_SIZE])
                        logging.info(f'Generated and saved {source_filename} in {args.source_dir}')
            else:
                logging.error("No image has been loaded - something has gone wrong somewhere!")
    except Exception as e:
        logging.error(f"An error occurred in a thread: {e}")
    finally:
        if conn and conn.isConnected():
            conn.close()
            logging.info(f"Thread finished. Connection closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--max_crosstalk', type=float, default=0.5,
                        help='Maximum relative crosstalk applied from source channel to bleed channel')
    parser.add_argument('-b', '--mixed_dir', help='Directory to save generated "bleed-through" images')
    parser.add_argument('-s', '--source_dir', help='Directory to save original "source" channel')
    parser.add_argument('-g', '--ground_truth_dir',
                        help='Directory to save "ground truth" images showing extent of bleed-through')
    parser.add_argument('-n', '--number_of_images', type=int, default=1,
                        help='Number of image sets to generate')
    parser.add_argument('-l', '--log_file', type=str, default='processing.log',
                        help='Path to the log file for detailed output')
    args = parser.parse_args()

    # Configure the logger to output to a file and not the console
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(args.log_file, mode='w')])

    # Let the user know where the detailed log can be found
    print(f"Detailed logs are being written to {args.log_file}")
    # --- Start the timer here ---
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
        futures = [executor.submit(worker_task, args, count) for count in work_items]

        # Use tqdm to create a progress bar for the completion of each worker task
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Workers completed"):
            try:
                future.result()
            except Exception as e:
                # This catches exceptions from the threads that are not handled within the worker_task function.
                logging.exception("An exception from a thread was propagated.")
    # --- End the timer and print the result ---
    end_time = time.time()
    total_time = end_time - start_time
    print("\nAll tasks completed. Please check the log file for details.")
    print(f"Total execution time: {total_time:.2f} seconds")
