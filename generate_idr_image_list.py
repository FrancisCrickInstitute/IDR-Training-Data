import argparse
import os
import random
import csv

from omero.gateway import BlitzGateway

MIN_SIZE = 256
DEFAULT_TARGET_X_DIM = 1024
DEFAULT_TARGET_Y_DIM = 1024

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_csv',
                    help='Path to the output CSV file where image details will be logged',
                    default='image_details.csv')
parser.add_argument('-n', '--number_of_images',
                    help='Number of image sets to log',
                    type=int, default=1)
parser.parse_args()

output_csv = parser.parse_args().output_csv
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


HOST = 'ws://idr.openmicroscopy.org/omero-ws'
conn = BlitzGateway('public', 'public', host=HOST, secure=True)
print(conn.connect())
conn.c.enableKeepAlive(60)

projects = list(conn.getObjects("Project"))
screens = list(conn.getObjects("Screen"))  # Get all screens

# Define the attribute and the value you're looking for
attribute_name = "Imaging Method"
# Define your search terms as a list
search_terms = ["fluorescence", "confocal"]

# Open the CSV file in write mode
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['Image ID', 'Image Name', 'Source Type', 'Project ID', 'Project Name',
                  'Dataset ID', 'Dataset Name', 'Screen ID', 'Screen Name',
                  'Plate ID', 'Plate Name', 'Well ID', 'Well Name',
                  'Size Z', 'Size C', 'Size T', 'Size Y', 'Size X',
                  'Selected Z Plane', 'Selected T Timepoint',
                  'Source Channel Index', 'Target Channel Index',
                  'Cropped Start X', 'Cropped Start Y', 'Cropped Size X', 'Cropped Size Y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for j in range(n_images):
        print(f'Obtaining image set {j} of {n_images}')
        found_suitable_image = False

        image_details = {
            'Project ID': '', 'Project Name': '', 'Dataset ID': '', 'Dataset Name': '',
            'Screen ID': '', 'Screen Name': '', 'Plate ID': '', 'Plate Name': '',
            'Well ID': '', 'Well Name': ''
        }  # Initialize all fields

        while not found_suitable_image:
            # Randomly choose between Project/Dataset and Screen/Plate/Well hierarchy
            if random.choice([True, False]) and projects:  # Prioritize Project if available
                source_type = 'Project/Dataset'
                print('Searching in Project/Dataset hierarchy...')
                random_project = random.choice(projects)
                print(f'Checking project {random_project.getName()}')
                kv_annotations = random_project.listAnnotations()
                for annotation in kv_annotations:
                    if hasattr(annotation, 'getMapValue'):
                        for key_value_pair in annotation.getMapValue():
                            key = key_value_pair.name
                            value = key_value_pair.value
                            if key == attribute_name:
                                print(f'{key}: {value}')
                                if any(term.lower() in value.lower() for term in search_terms):
                                    datasets = list(random_project.listChildren())
                                    if not datasets:
                                        continue
                                    random_dataset = random.choice(datasets)
                                    images = list(random_dataset.listChildren())
                                    if not images:
                                        continue
                                    random_image = random.choice(images)
                                    pixels = random_image.getPrimaryPixels()
                                    if pixels.getSizeC() > 1 and pixels.getSizeX() > MIN_SIZE and pixels.getSizeY() > MIN_SIZE:
                                        found_suitable_image = True
                                        image_details['Source Type'] = source_type
                                        image_details['Image ID'] = random_image.getId()
                                        image_details['Image Name'] = random_image.getName()
                                        image_details['Project ID'] = random_project.getId()
                                        image_details['Project Name'] = random_project.getName()
                                        image_details['Dataset ID'] = random_dataset.getId()
                                        image_details['Dataset Name'] = random_dataset.getName()
                                        break
                if found_suitable_image:
                    parent1_obj = random_project
                    parent2_obj = random_dataset
                    selected_image_obj = random_image

            elif screens:  # If no project found or randomly chosen, try Screen
                source_type = 'Screen/Plate/Well'
                print('Searching in Screen/Plate/Well hierarchy...')
                random_screen = random.choice(screens)
                print(f'Checking screen {random_screen.getName()}')
                kv_annotations = random_screen.listAnnotations()
                for annotation in kv_annotations:
                    if hasattr(annotation, 'getMapValue'):
                        for key_value_pair in annotation.getMapValue():
                            key = key_value_pair.name
                            value = key_value_pair.value
                            if key == attribute_name:
                                print(f'{key}: {value}')
                                if any(term.lower() in value.lower() for term in search_terms):
                                    plates = list(random_screen.listChildren())
                                    if not plates:
                                        continue
                                    random_plate = random.choice(plates)
                                    wells = list(random_plate.listChildren())
                                    if not wells:
                                        continue
                                    random_well = random.choice(wells)
                                    # Get images from the well. Note: getWellSamples() returns WellSample objects
                                    # which can then be used to get the Image.
                                    well_samples = list(random_well.listChildren())
                                    images_in_well = []
                                    for ws in well_samples:
                                        img = ws.getImage()
                                        if img:
                                            images_in_well.append(img)

                                    if not images_in_well:
                                        continue
                                    random_image = random.choice(images_in_well)
                                    pixels = random_image.getPrimaryPixels()
                                    if pixels.getSizeC() > 1 and pixels.getSizeX() > MIN_SIZE and pixels.getSizeY() > MIN_SIZE:
                                        found_suitable_image = True
                                        image_details['Source Type'] = source_type
                                        image_details['Image ID'] = random_image.getId()
                                        image_details['Image Name'] = random_image.getName()
                                        image_details['Screen ID'] = random_screen.getId()
                                        image_details['Screen Name'] = random_screen.getName()
                                        image_details['Plate ID'] = random_plate.getId()
                                        image_details['Plate Name'] = random_plate.getName()
                                        image_details['Well ID'] = random_well.getId()
                                        image_details['Well Name'] = random_well.getName()
                                        break
                if found_suitable_image:
                    parent1_obj = random_screen
                    parent2_obj = random_plate
                    parent3_obj = random_well
                    selected_image_obj = random_image

            if not found_suitable_image:  # If neither found or no suitable image in chosen path, continue the loop
                print("No suitable image found in the current hierarchy. Trying again...")
                continue  # Go back to the start of the while loop to pick another project/screen

        pixels = selected_image_obj.getPrimaryPixels()  # Get pixels object after finding a suitable image
        image_details['Size Z'] = pixels.getSizeZ()
        image_details['Size C'] = pixels.getSizeC()
        image_details['Size T'] = pixels.getSizeT()
        image_details['Size Y'] = pixels.getSizeY()
        image_details['Size X'] = pixels.getSizeX()

        if image_details['Source Type'] == 'Project/Dataset':
            print_obj(parent1_obj)
            print_obj(parent2_obj)
        elif image_details['Source Type'] == 'Screen/Plate/Well':
            print_obj(parent1_obj)
            print_obj(parent2_obj)
            print_obj(parent3_obj)  # Print well
        print_obj(selected_image_obj)

        # Replicate cropping logic to log the region
        size_y = image_details['Size Y']
        size_x = image_details['Size X']

        cropped_size_y = min(DEFAULT_TARGET_Y_DIM, size_y)
        cropped_size_x = min(DEFAULT_TARGET_X_DIM, size_x)

        start_y = random.randint(0, size_y - cropped_size_y) if size_y > DEFAULT_TARGET_Y_DIM else 0
        start_x = random.randint(0, size_x - cropped_size_x) if size_x > DEFAULT_TARGET_X_DIM else 0

        image_details['Cropped Start X'] = start_x
        image_details['Cropped Start Y'] = start_y
        image_details['Cropped Size X'] = cropped_size_x
        image_details['Cropped Size Y'] = cropped_size_y

        # Randomly select Z and T planes and channels for logging
        selected_z_plane = random.randint(0, image_details['Size Z'] - 1)
        selected_t_timepoint = random.randint(0, image_details['Size T'] - 1)
        source_channel = random.choice(range(image_details['Size C']))
        target_channel = source_channel
        while source_channel == target_channel:
            target_channel = random.choice(range(image_details['Size C']))

        image_details['Selected Z Plane'] = selected_z_plane
        image_details['Selected T Timepoint'] = selected_t_timepoint
        image_details['Source Channel Index'] = source_channel
        image_details['Target Channel Index'] = target_channel

        print(f'Source Channel: {source_channel}')
        print(f'Target Channel: {target_channel}')
        print(f"Cropped Region: X={start_x}-{start_x + cropped_size_x}, Y={start_y}-{start_y + cropped_size_y}")

        writer.writerow(image_details)
        print(f"Logged details for image {selected_image_obj.getId()} to {output_csv}")

conn.close()