import argparse
import os
import random

import imageio.v3 as iio  # Using imageio v3 for modern API
import numpy as np
from omero.gateway import BlitzGateway
from skimage.transform import resize

MIN_SIZE = 256
NORM_COEFF = np.power(2, 16)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--max_crosstalk',
                    help='Maximum relative crosstalk applied from source channel to bleed channel',
                    type=float, default=0.5)
parser.add_argument('-b', '--mixed_dir', help='Directory to save generated "bleed-through" images')
parser.add_argument('-s', '--source_dir', help='Directory to save original "source" channel')
parser.add_argument('-g', '--ground_truth_dir',
                    help='Directory to save "ground truth" images showing extent of bleed-through')
parser.add_argument('-n', '--number_of_images',
                    help='Number of image sets to generate',
                    type=int, default=1)
parser.parse_args()

max_crosstalk = parser.parse_args().max_crosstalk
mixed_dir = parser.parse_args().mixed_dir
source_dir = parser.parse_args().source_dir
ground_truth_dir = parser.parse_args().ground_truth_dir
n_images = parser.parse_args().number_of_images


def print_obj(obj, indent=0):
    """
    Helper method to display info about OMERO objects.
    Not all objects will have a "name" or owner field.
    """
    print("""%s%s:%s  Name:"%s" (owner=%s)""" % (
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
        print(
            f"Warning: Image {image.getName()} has invalid dimensions (Z:{size_z}, C:{size_c}, T:{size_t}). Cannot load planes.")
        return None

    # Randomly select Z and T planes
    selected_z_plane = random.randint(0, size_z - 1)
    selected_t_timepoint = random.randint(0, size_t - 1)

    s = "t:%s c:%s z:%s y:%s x:%s (selected t:%s z:%s)" % \
        (size_t, size_c, size_z, size_y, size_x, selected_t_timepoint, selected_z_plane)
    print(s)

    # --- Determine Cropping Dimensions and Start Points ---
    # Ensure cropped dimensions don't exceed original dimensions
    cropped_size_y = min(target_y_dim, size_y)
    cropped_size_x = min(target_x_dim, size_x)

    # Calculate random start_y/start_x if original dim is larger than target
    start_y = random.randint(0, size_y - cropped_size_y) if size_y > target_y_dim else 0
    start_x = random.randint(0, size_x - cropped_size_x) if size_x > target_x_dim else 0

    # --- Efficiently Load Only the Cropped Region ---
    loaded_planes = []
    print(
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
        print(f"Error loading cropped planes for image {image.getName()}: {e}")
        return None

    # Reshape to (1, size_c, 1, cropped_size_y, cropped_size_x)
    final_shape = (1, size_c, 1, cropped_size_y, cropped_size_x)
    return np.reshape(all_channels_cropped_data, newshape=final_shape)


def save_images(image: np.ndarray, filename: str, new_shape=None):
    """
    Saves a NumPy array as a TIFF image.
    Resizes the image if new_shape is specified.
    Handles appropriate data type conversion for saving.

    Parameters:
    - image: NumPy array representing the image.
    - filename: String representing the filename to save the image as.
    - new_shape: Tuple representing the new shape (height, width) for the image.
    """
    if new_shape is not None:
        # Resize the image with interpolation
        image = resize(image, new_shape, preserve_range=True)

    # Imageio can handle float32 directly for TIFF.
    iio.imwrite(filename, image, extension='.tif')
    print(f"Saved {filename}")


def generate_crosstalk_data(pure_target_channel: np.ndarray, pure_source_channel: np.ndarray,
                            crosstalk_coefficient: float) -> tuple[np.ndarray, float]:
    if not (0.0 <= crosstalk_coefficient <= 1.0):
        print("Warning: Crosstalk coefficient is typically between 0 and 1.")

    if pure_target_channel.shape != pure_source_channel.shape:
        raise ValueError("Pure target and source channel images must have the same shape.")

    # Ensure inputs are float to allow for scaling and addition
    pure_target_channel = pure_target_channel.astype(np.float64)
    pure_source_channel = pure_source_channel.astype(np.float64)

    target_sum = float(np.sum(pure_target_channel))
    source_sum = float(np.sum(pure_source_channel))

    alpha = (crosstalk_coefficient * target_sum) / (source_sum * (1.0 - crosstalk_coefficient))

    # Calculate the bleed-through signal
    bleed_through_signal = alpha * pure_source_channel

    # Generate the mixed target channel image
    mixed_target_channel = pure_target_channel + bleed_through_signal

    # Rescale the entire image based on its actual min/max
    current_max = np.max(mixed_target_channel)

    if current_max == 0:  # Handle uniform image to avoid division by zero
        normalized_mixed_target_channel = np.zeros_like(mixed_target_channel)
    else:
        # Scale to 0 to NORM_COEFF range
        normalized_mixed_target_channel = mixed_target_channel * NORM_COEFF / current_max

    sum_mixed_target_channel = float(np.sum(mixed_target_channel))

    if sum_mixed_target_channel > 0.0:
        bleedthrough_proportion = float(np.sum(bleed_through_signal)) / sum_mixed_target_channel
    else:
        bleedthrough_proportion = 0.0

    return normalized_mixed_target_channel / NORM_COEFF, bleedthrough_proportion  # Assuming you still want 0-1 float output


HOST = 'ws://idr.openmicroscopy.org/omero-ws'
conn = BlitzGateway('public', 'public', host=HOST, secure=True)
print(conn.connect())
conn.c.enableKeepAlive(60)

projects = list(conn.getObjects("Project"))

# Define the attribute and the value you're looking for
attribute_name = "Imaging Method"
# Define your search terms as a list
search_terms = ["fluorescence", "confocal"]

# Set to keep track of processed image IDs
processed_image_ids = set()

for j in range(n_images):
    print(f'Obtaining image set {j} of {n_images}')
    foundProject = False

    while not foundProject:
        random_project = random.choice(projects)
        print(f'Checking project {random_project.getName()}')
        kv_annotations = random_project.listAnnotations()  # or specific namespace if known
        for annotation in kv_annotations:
            if hasattr(annotation, 'getMapValue'):  # check if it's a MapAnnotation
                for key_value_pair in annotation.getMapValue():
                    key = key_value_pair.name
                    value = key_value_pair.value
                    if key == attribute_name:
                        print(f'{key}: {value}')
                        if any(term.lower() in value.lower() for term in search_terms):
                            datasets = list(random_project.listChildren())
                            random_dataset = random.choice(datasets)
                            images = list(random_dataset.listChildren())
                            random_image = random.choice(images)
                            if (random_image.getId() not in processed_image_ids and
                                    random_image.getPrimaryPixels().getSizeC() > 1 and
                                    random_image.getPrimaryPixels().getSizeX() > MIN_SIZE and
                                    random_image.getPrimaryPixels().getSizeY() > MIN_SIZE):
                                foundProject = True
                                processed_image_ids.add(random_image.getId())
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

        print(f'Source Channel: {source_channel}')
        print(f'Target Channel: {target_channel}')

        # Define a set of crosstalk coefficients to generate diverse data
        crosstalk_coefficients = [random.random() * max_crosstalk]  # Include no crosstalk (0.0)

        for j, alpha in enumerate(crosstalk_coefficients):
            mixed_image, crosstalk_proportion = generate_crosstalk_data(
                pure_target_channel=data[0, target_channel, 0],
                pure_source_channel=data[0, source_channel, 0],
                crosstalk_coefficient=alpha
            )

            # Generate unique filenames
            mixed_filename = os.path.join(mixed_dir,
                                          f"image_{random_image.getId()}_alpha_{crosstalk_proportion:.2f}_mixed.tif")
            source_filename = os.path.join(source_dir,
                                           f"image_{random_image.getId()}_alpha_{crosstalk_proportion:.2f}_source.tif")

            # Save the generated images
            save_images(mixed_image, mixed_filename, new_shape=[MIN_SIZE, MIN_SIZE])
            print(f'Generated and saved {mixed_filename} in {mixed_dir}')
            save_images(data[0, source_channel, 0] / NORM_COEFF, source_filename, new_shape=[MIN_SIZE, MIN_SIZE])
            print(f'Generated and saved {source_filename} in {source_dir}')
    else:
        print("No image has been loaded - something has gone wrong somewhere!")

conn.close()
