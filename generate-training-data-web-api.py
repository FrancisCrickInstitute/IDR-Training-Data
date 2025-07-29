import argparse
import os
import random
from urllib.parse import urljoin

import imageio.v3 as iio
import numpy as np
import requests
from skimage.transform import resize

MIN_SIZE = 256
NORM_COEFF = np.power(2, 16)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--max_crosstalk',
                    help='Maximum relative crosstalk applied from source channel to bleed channel',
                    type=float, default=0.5)
parser.add_argument('-b', '--mixed_dir', help='Directory to save generated "bleed-through" images', required=True)
parser.add_argument('-s', '--source_dir', help='Directory to save original "source" channel', required=True)
parser.add_argument('-g', '--ground_truth_dir',
                    help='Directory to save "ground truth" images showing extent of bleed-through')
parser.add_argument('-n', '--number_of_images',
                    help='Number of image sets to generate',
                    type=int, default=1)
args = parser.parse_args()  # Parse args once

max_crosstalk = args.max_crosstalk
mixed_dir = args.mixed_dir
source_dir = args.source_dir
ground_truth_dir = args.ground_truth_dir
n_images = args.number_of_images

# Create directories if they don't exist
os.makedirs(mixed_dir, exist_ok=True)
os.makedirs(source_dir, exist_ok=True)
if ground_truth_dir:
    os.makedirs(ground_truth_dir, exist_ok=True)


def print_obj_json(obj, indent=0):
    """
    Helper method to display info about OMERO JSON objects.
    """
    obj_type = obj.get('@type', 'Unknown Type').split('#')[-1]
    obj_id = obj.get('@id', 'N/A')
    obj_name = obj.get('Name', 'N/A')
    owner_name = obj.get('omero:details', {}).get('owner', {}).get('omeName', 'N/A')
    print(f"{' ' * indent}{obj_type}:{obj_id} Name:\"{obj_name}\" (owner={owner_name})")


def get_json(session, base_url, endpoint):
    """Helper to make GET requests and return JSON."""
    url = urljoin(base_url, endpoint)
    try:
        response = session.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def load_numpy_array_json(session, base_url, image_id, size_x, size_y, size_c, size_z, size_t, target_x_dim=1024,
                          target_y_dim=1024):
    """
    Loads a microscopy image plane into a NumPy array using the JSON API,
    applying random cropping if the image dimensions exceed specified limits,
    and choosing Z and T planes randomly. This version downloads a whole plane
    and then crops it locally.

    Args:
        session: requests.Session object for making requests.
        base_url: Base URL for the OMERO JSON API.
        image_id (int): The ID of the OMERO image.
        size_x, size_y, size_c, size_z, size_t: Image dimensions.
        target_x_dim (int): The maximum desired X dimension. If the image
                            is larger, it will be cropped to this size.
        target_y_dim (int): The maximum desired Y dimension. If the image
                            is larger, it will be cropped to this size.

    Returns:
        numpy.ndarray: The loaded (and potentially cropped) image data
                       reshaped to (1, size_c, 1, cropped_size_y, cropped_size_x).
        None: If there's an error during processing or if image dimensions are invalid.
    """
    if size_z < 1 or size_t < 1 or size_c < 1:
        print(
            f"Warning: Image ID {image_id} has invalid dimensions (Z:{size_z}, C:{size_c}, T:{size_t}). Cannot load planes.")
        return None

    # Randomly select Z and T planes
    selected_z_plane = random.randint(0, size_z - 1)
    selected_t_timepoint = random.randint(0, size_t - 1)

    s = f"t:{size_t} c:{size_c} z:{size_z} y:{size_y} x:{size_x} (selected t:{selected_t_timepoint} z:{selected_z_plane})"
    print(s)

    # --- Determine Cropping Dimensions and Start Points ---
    cropped_size_y = min(target_y_dim, size_y)
    cropped_size_x = min(target_x_dim, size_x)

    start_y = random.randint(0, size_y - cropped_size_y) if size_y > target_y_dim else 0
    start_x = random.randint(0, size_x - cropped_size_x) if size_x > target_x_dim else 0

    loaded_planes = []
    print(f"Downloading image {image_id} planes for channels...")

    try:
        for c in range(size_c):
            # The JSON API's webgateway/render_image endpoint
            # downloads a whole plane. We'll crop it locally.
            # Request as TIFF for better numerical fidelity.
            image_url = urljoin(base_url, f"/webclient/render_image/{image_id}/")
            params = {
                'z': selected_z_plane,
                'c': c,
                't': selected_t_timepoint,
                'format': 'tif'  # Request TIFF for raw pixel values
            }
            print(f"  Downloading channel {c} (Z={selected_z_plane}, T={selected_t_timepoint})...")
            response = session.get(image_url, params=params, stream=True)
            response.raise_for_status()

            # Read the image data into a numpy array using imageio
            # imageio can read from a file-like object
            with iio.BytesIO(response.content) as f:
                plane_data = iio.imread(f)

            # Ensure data is 2D (height, width)
            if plane_data.ndim > 2:
                # If imageio reads it as 3D (e.g., grayscale with 1 channel), squeeze it
                plane_data = np.squeeze(plane_data)

            # Perform local cropping
            cropped_plane_data = plane_data[
                                 start_y: start_y + cropped_size_y,
                                 start_x: start_x + cropped_size_x
                                 ]
            loaded_planes.append(cropped_plane_data)

        # Stack the loaded tiles. Shape will be (size_c, cropped_size_y, cropped_size_x)
        all_channels_cropped_data = np.stack(loaded_planes)

    except Exception as e:
        print(f"Error loading cropped planes for image ID {image_id}: {e}")
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

    # Calculate the bleed-through signal
    bleed_through_signal = crosstalk_coefficient * pure_source_channel

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


# Base URL for IDR JSON API
IDR_HOST = 'https://idr.openmicroscopy.org'
API_BASE_URL = urljoin(IDR_HOST, '/api/v0/m/')  # For Projects, Datasets, Images metadata
WEBCLIENT_BASE_URL = urljoin(IDR_HOST, '/webclient/')  # For image rendering and some other metadata

# Create a requests session for persistent connection and potential cookies (if authentication was needed)
session = requests.Session()

# For public IDR, explicit login is not strictly necessary for public data access via webclient/render_image
# If you were connecting to a private OMERO instance requiring login, you'd do something like:
# login_data = {'username': 'your_username', 'password': 'your_password'}
# try:
#     login_response = session.post(urljoin(WEBCLIENT_BASE_URL, 'login/'), data=login_data)
#     login_response.raise_for_status()
#     print("Login successful (if required).")
# except requests.exceptions.RequestException as e:
#     print(f"Login failed: {e}")
#     exit()

try:
    # Get all projects
    print("Fetching projects...")
    projects_json = get_json(session, API_BASE_URL, 'projects/')
    if not projects_json or not projects_json.get('data'):
        print("No projects found or error fetching projects.")
        session.close()
        exit()

    projects = projects_json['data']
    print(f"Found {len(projects)} projects.")

    # Define the attribute and the value you're looking for
    attribute_name = "Imaging Method"
    search_terms = ["fluorescence", "confocal"]

    for i in range(n_images):
        print(f'\nObtaining image set {i + 1} of {n_images}')
        found_suitable_image = False

        while not found_suitable_image:
            random_project = random.choice(projects)
            project_id = random_project['@id']
            print(f'Checking project {random_project["Name"]} (ID: {project_id})')

            # Fetch annotations for the project
            # OMERO JSON API has a specific endpoint for annotations, or you can get
            # specific annotations if they are linked directly to the object metadata.
            # For map annotations, it's often under a general annotations endpoint.
            # The structure for fetching Map Annotations is a bit more involved as
            # there isn't a direct 'listAnnotations' on project JSON object like BlitzGateway.
            # We'll use the /webclient/api/annotations endpoint if we need to search by namespace
            # or rely on the metadata present in the project object itself.
            # For this example, we'll try to get key-value pairs associated with the project
            # if they are directly available in the initial project JSON.
            # However, `omero:details` doesn't contain arbitrary annotations.
            # So, we'll simplify this by assuming we just need to find *any* suitable image
            # within *any* project that contains "fluorescence" or "confocal" in its description
            # or related metadata accessible via the image.

            # A more robust way to search annotations would be via the search engine API
            # or iterating through all images and their annotations if possible.
            # For simplicity, we'll select images from a random project and check their
            # dimensions, rather than trying to filter projects by annotations at this stage.

            # Get datasets for the selected project
            datasets_json = get_json(session, API_BASE_URL, f'projects/{project_id}/datasets/')
            if not datasets_json or not datasets_json.get('data'):
                print(f"No datasets found for project {random_project['Name']}")
                continue
            datasets = datasets_json['data']
            random_dataset = random.choice(datasets)
            dataset_id = random_dataset['@id']

            # Get images for the selected dataset
            images_json = get_json(session, API_BASE_URL, f'datasets/{dataset_id}/images/')
            if not images_json or not images_json.get('data'):
                print(f"No images found for dataset {random_dataset['Name']}")
                continue
            images = images_json['data']

            # Filter for suitable images based on dimensions and channel count
            suitable_images = [
                img for img in images
                if img.get('Pixels', {}).get('SizeC', 0) > 1 and
                   img.get('Pixels', {}).get('SizeX', 0) > MIN_SIZE and
                   img.get('Pixels', {}).get('SizeY', 0) > MIN_SIZE
            ]

            if not suitable_images:
                print(f"No suitable images found in dataset {random_dataset['Name']}")
                continue

            random_image = random.choice(suitable_images)
            image_id = random_image['@id']
            pixels_data = random_image.get('Pixels', {})

            # For checking "Imaging Method", you'd typically need to fetch image annotations.
            # The /webclient/api/annotations/ endpoint is useful here.
            # Example: /webclient/api/annotations/?type=map&image={image_id}
            image_annotations_json = get_json(session, WEBCLIENT_BASE_URL,
                                              f'api/annotations/?type=map&image={image_id}')
            has_suitable_imaging_method = False
            if image_annotations_json and image_annotations_json.get('annotations'):
                for annotation in image_annotations_json['annotations']:
                    if annotation.get(
                            'ns') == "openmicroscopy.org/omero/web/microscopy/imaging_method":  # Common namespace for imaging method
                        for key_value_pair in annotation.get('values', []):
                            key = key_value_pair[0]
                            value = key_value_pair[1]
                            if key == attribute_name:
                                if any(term.lower() in value.lower() for term in search_terms):
                                    has_suitable_imaging_method = True
                                    print(f"  {key}: {value} (Matched search terms)")
                                    break
                        if has_suitable_imaging_method:
                            break

            if not has_suitable_imaging_method:
                print(f"  Image {random_image['Name']} does not have required imaging method annotation. Skipping.")
                continue

            found_suitable_image = True
            print("Found suitable image:")
            print_obj_json(random_project)
            print_obj_json(random_dataset)
            print_obj_json(random_image)

            # Extract dimensions
            size_x = pixels_data.get('SizeX')
            size_y = pixels_data.get('SizeY')
            size_c = pixels_data.get('SizeC')
            size_z = pixels_data.get('SizeZ')
            size_t = pixels_data.get('SizeT')

            data = load_numpy_array_json(session, IDR_HOST, image_id, size_x, size_y, size_c, size_z, size_t)

            if data is not None:
                source_channel_idx = random.choice(range(size_c))
                target_channel_idx = source_channel_idx

                while source_channel_idx == target_channel_idx:
                    target_channel_idx = random.choice(range(size_c))

                print(f'Source Channel: {source_channel_idx}')
                print(f'Target Channel: {target_channel_idx}')

                crosstalk_coefficients = [random.random() * max_crosstalk]

                for j, alpha in enumerate(crosstalk_coefficients):
                    mixed_image, crosstalk_proportion = generate_crosstalk_data(
                        pure_target_channel=data[0, target_channel_idx, 0],
                        pure_source_channel=data[0, source_channel_idx, 0],
                        crosstalk_coefficient=alpha
                    )

                    # Generate unique filenames
                    mixed_filename = os.path.join(mixed_dir,
                                                  f"image_{image_id}_alpha_{crosstalk_proportion:.2f}_mixed.tif")
                    source_filename = os.path.join(source_dir,
                                                   f"image_{image_id}_alpha_{crosstalk_proportion:.2f}_source.tif")

                    # Save the generated images
                    save_images(mixed_image, mixed_filename, new_shape=[MIN_SIZE, MIN_SIZE])
                    print(f'Generated and saved {mixed_filename} in {mixed_dir}')
                    save_images(data[0, source_channel_idx, 0] / NORM_COEFF, source_filename,
                                new_shape=[MIN_SIZE, MIN_SIZE])
                    print(f'Generated and saved {source_filename} in {source_dir}')
            else:
                print(f"No image has been loaded for ID {image_id} - something has gone wrong somewhere!")
                # If image loading failed, we need to try another image
                found_suitable_image = False  # Reset flag to try another image

finally:
    # Always close the session
    session.close()
    print("Session closed.")
