import argparse
import os
import random

import imageio.v3 as iio  # Using imageio v3 for modern API
import numpy as np
from omero.gateway import BlitzGateway

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--max_crosstalk',
                    help='Maximum relative crosstalk applied from source channel to bleed channel',
                    type=float, default=0.5)
parser.add_argument('-b', '--mixed_dir', help='Directory to save generated "bleed-through" images')
parser.add_argument('-s', '--source_dir', help='Directory to save original "source" channel')
parser.add_argument('-g', '--ground_truth_dir',
                    help='Directory to save "ground truth" images showing extent of bleed-through')
parser.parse_args()

max_crosstalk = parser.parse_args().max_crosstalk
mixed_dir = parser.parse_args().mixed_dir
source_dir = parser.parse_args().source_dir
ground_truth_dir = parser.parse_args().ground_truth_dir

from datetime import datetime


def add_timestamp_to_print(input_string):
    # Get the current date and time
    current_time = datetime.now()
    # Format the current time as a string
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # Concatenate the timestamp with the input string
    print(f"{timestamp} {input_string}")


def print_obj(obj, indent=0):
    """
    Helper method to display info about OMERO objects.
    Not all objects will have a "name" or owner field.
    """
    add_timestamp_to_print("""%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        obj.OMERO_CLASS,
        obj.getId(),
        obj.getName(),
        obj.getOwnerOmeName()))


def load_numpy_array(image, target_x_dim=1024, target_y_dim=1024):
    """
    Loads a microscopy image into a NumPy array, applying random cropping
    if the image dimensions exceed specified limits, and choosing Z and T
    planes randomly. This version loads only the cropped region efficiently.

    Args:
        image: An OMERO image object (e.g., omero.gateway.ImageWrapper)
               from which to load pixel data and metadata.
        target_x_dim (int): The maximum desired X dimension. If the image
                            is larger, it will be cropped to this size.
        target_y_dim (int): The maximum desired Y dimension. If the image
                            is larger, it will be cropped to this size.

    Returns:
        numpy.ndarray: The loaded (and potentially cropped) image data
                       reshaped to (1, size_c, 1, cropped_size_y, cropped_size_x).
        None: If there's an error during processing or if image dimensions are invalid.
    """
    pixels = image.getPrimaryPixels()

    size_z = pixels.getSizeZ()
    size_c = pixels.getSizeC()
    size_t = pixels.getSizeT()
    size_y = pixels.getSizeY()
    size_x = pixels.getSizeX()

    # Ensure dimensions are valid for loading
    if size_z < 1 or size_t < 1 or size_c < 1:
        add_timestamp_to_print(
            f"Warning: Image {image.getName()} has invalid dimensions (Z:{size_z}, C:{size_c}, T:{size_t}). Cannot load planes.")
        return None

    # Randomly select Z and T planes
    selected_z_plane = random.randint(0, size_z - 1)
    selected_t_timepoint = random.randint(0, size_t - 1)

    s = "t:%s c:%s z:%s y:%s x:%s (selected t:%s z:%s)" % \
        (size_t, size_c, size_z, size_y, size_x, selected_t_timepoint, selected_z_plane)
    add_timestamp_to_print(s)

    # --- Determine Cropping Dimensions and Start Points ---
    # Ensure cropped dimensions don't exceed original dimensions
    cropped_size_y = min(target_y_dim, size_y)
    cropped_size_x = min(target_x_dim, size_x)

    # Calculate random start_y/start_x if original dim is larger than target
    start_y = random.randint(0, size_y - cropped_size_y) if size_y > target_y_dim else 0
    start_x = random.randint(0, size_x - cropped_size_x) if size_x > target_x_dim else 0

    # --- Efficiently Load Only the Cropped Region ---
    loaded_planes = []
    add_timestamp_to_print(
        f"Downloading image {image.getName()} (cropped region: X={start_x}-{start_x + cropped_size_x}, Y={start_y}-{start_y + cropped_size_y})")
    try:
        for c in range(size_c):
            # --- FIX: Pass Z, C, T, and the tile coordinates as a single tuple ---
            # This signature matches the argument count constraints and the pattern
            # implied by your getTiles example for a single tile.
            tile_coords_tuple = (start_x, start_y, cropped_size_x, cropped_size_y)

            # The assumed getTile() signature is now:
            # getTile(z, c, t, (x, y, width, height))
            plane_data = pixels.getTile(
                selected_z_plane, c, selected_t_timepoint, tile_coords_tuple
            )
            loaded_planes.append(plane_data)

        # Stack the loaded tiles. Shape will be (size_c, cropped_size_y, cropped_size_x)
        all_channels_cropped_data = np.stack(loaded_planes)

    except Exception as e:
        add_timestamp_to_print(f"Error loading cropped planes for image {image.getName()}: {e}")
        return None

    # Reshape to (1, size_c, 1, cropped_size_y, cropped_size_x)
    final_shape = (1, size_c, 1, cropped_size_y, cropped_size_x)
    return np.reshape(all_channels_cropped_data, newshape=final_shape)


def save_images(image: np.ndarray, filename: str):
    """
    Saves a NumPy array as a TIFF image.
    Handles appropriate data type conversion for saving.
    """
    # Imageio can handle float32 directly for TIFF.
    iio.imwrite(filename, image, extension='.tif')
    add_timestamp_to_print(f"Saved {filename}")


def generate_crosstalk_data(pure_target_channel: np.ndarray, pure_source_channel: np.ndarray,
                            crosstalk_coefficient: float) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= crosstalk_coefficient <= 1.0):
        add_timestamp_to_print("Warning: Crosstalk coefficient is typically between 0 and 1.")

    if pure_target_channel.shape != pure_source_channel.shape:
        raise ValueError("Pure target and source channel images must have the same shape.")

    # Ensure inputs are float to allow for scaling and addition
    pure_target_channel = pure_target_channel.astype(np.float64)
    pure_source_channel = pure_source_channel.astype(np.float64)

    # Calculate the bleed-through signal
    bleed_through_signal = crosstalk_coefficient * pure_source_channel

    # Generate the mixed target channel image
    mixed_target_channel = pure_target_channel + bleed_through_signal

    # The ground truth crosstalk map is simply the bleed-through signal itself
    ground_truth_crosstalk_map = bleed_through_signal

    return mixed_target_channel, ground_truth_crosstalk_map


HOST = 'ws://idr.openmicroscopy.org/omero-ws'
add_timestamp_to_print(f'Attempting to connect to {HOST}')
conn = BlitzGateway('public', 'public', host=HOST, secure=True)
if (conn.connect()):
    add_timestamp_to_print(f'Connected to {HOST}')
else:
    add_timestamp_to_print(f'Failed to connect to {HOST}')
    exit(1)
conn.c.enableKeepAlive(60)

add_timestamp_to_print(f'Getting project list from {HOST}')
projects = list(conn.getObjects("Project"))

# Define the attribute and the value you're looking for
attribute_name = "Imaging Method"
# Define your search terms as a list
search_terms = ["fluorescence", "confocal"]

foundProject = False

while not foundProject:
    random_project = random.choice(projects)
    add_timestamp_to_print(f'Checking project {random_project.getName()}')
    kv_annotations = random_project.listAnnotations()  # or specific namespace if known
    for annotation in kv_annotations:
        if hasattr(annotation, 'getMapValue'):  # check if it's a MapAnnotation
            for key_value_pair in annotation.getMapValue():
                key = key_value_pair.name
                value = key_value_pair.value
                if key == attribute_name:
                    add_timestamp_to_print(f'{key}: {value}')
                    if any(term.lower() in value.lower() for term in search_terms):
                        datasets = list(random_project.listChildren())
                        random_dataset = random.choice(datasets)
                        images = list(random_dataset.listChildren())
                        random_image = random.choice(images)
                        if random_image.getPrimaryPixels().getSizeC() > 1:
                            foundProject = True
                            break

print_obj(random_project)
print_obj(random_dataset)
print_obj(random_image)

data = load_numpy_array(random_image)

if data is not None:
    source_channel = random.choice(range(np.size(data[0], 0)))
    target_channel = source_channel

    while source_channel == target_channel:
        target_channel = random.choice(range(np.size(data[0], 0)))

    add_timestamp_to_print(f'Source Channel: {source_channel}')
    add_timestamp_to_print(f'Target Channel: {target_channel}')

    # Define a set of crosstalk coefficients to generate diverse data
    crosstalk_coefficients = [0.0, random.random() * max_crosstalk]  # Include no crosstalk (0.0)

    for i, alpha in enumerate(crosstalk_coefficients):
        mixed_image, true_crosstalk_map = generate_crosstalk_data(
            pure_target_channel=data[0, target_channel, 0],
            pure_source_channel=data[0, source_channel, 0],
            crosstalk_coefficient=alpha
        )

        # Generate unique filenames
        mixed_filename = os.path.join(mixed_dir, f"image_{random_image.getId()}_alpha_{alpha:.2f}_mixed.tif")
        ground_truth_filename = os.path.join(ground_truth_dir,
                                             f"image_{random_image.getId()}_alpha_{alpha:.2f}_ground_truth.tif")
        source_filename = os.path.join(source_dir, f"image_{random_image.getId()}_alpha_{alpha:.2f}_source.tif")

        # Save the generated images
        save_images(mixed_image, mixed_filename)
        add_timestamp_to_print(f'Generated and saved {mixed_filename} in {mixed_dir}')
        save_images(true_crosstalk_map, ground_truth_filename)
        add_timestamp_to_print(f'Generated and saved {ground_truth_filename} in {ground_truth_dir}')
        save_images(data[0, source_channel, 0], source_filename)
        add_timestamp_to_print(f'Generated and saved {source_filename} in {source_dir}')
else:
    add_timestamp_to_print("No image has been loaded - something has gone wrong somewhere!")

conn.close()
