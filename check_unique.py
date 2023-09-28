import os
import numpy as np
from PIL import Image
import cv2

def get_unique_values_in_folder(folder_path):
    unique_values_per_file = {}

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Open the image using PIL and convert to NumPy array
        numpy_image = np.array(Image.open(file_path))
        print("Shape",numpy_image.shape)
        # numpy_image = np.array(Image.open(file_path))

        # Get the unique values in the NumPy array
        unique_values = np.unique(numpy_image)

        # Save the unique values in the dictionary with the filename as the key
        unique_values_per_file[filename] = unique_values

    return unique_values_per_file

def main():
    folder_path_1 = "./image-parse-test"  # Replace with the actual path to the first folder
    folder_path_2 = "./from_here"  # Replace with the actual path to the second folder

    # Get unique values for each file in the first folder
    unique_values_folder1 = get_unique_values_in_folder(folder_path_1)

    # Get unique values for each file in the second folder
    unique_values_folder2 = get_unique_values_in_folder(folder_path_2)

    # Print the unique values for each file in the first folder
    print("Unique Values in Folder {}:".format(folder_path_1))
    for filename, unique_values in unique_values_folder1.items():
        print(f"{filename}: {unique_values}")

    # Print the unique values for each file in the second folder
    print("\nUnique Values in Folder {}:".format(folder_path_2))
    for filename, unique_values in unique_values_folder2.items():
        print(f"{filename}: {unique_values}")

if __name__ == "__main__":
    main()
