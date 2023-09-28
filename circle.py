import cv2
import numpy as np
from scipy import ndimage
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os

colormap = [
    (0, 0, 0),
    (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
    (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
    (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
    (1, 0, 0.75), (0, 0.5, 0.5), (0.25, 0.5, 0.5), (1, 0, 0),
    (1, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0)
]

color_map= ListedColormap(colormap)

fig, ax = plt.subplots(1, 1)

def manual_maximum_filter(input_array, edge,size ):
    height, width = input_array.shape
    radius = size // 2

    max_filtered = np.zeros_like(input_array)

    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            if edge[y,x] > 0:
                neighborhood = input_array[y - radius:y + radius + 1, x - radius:x + radius + 1]
                unique_values, counts = np.unique(neighborhood, return_counts=True)
                most_common_value = unique_values[np.argmax(counts)]
                # print(unique_values)
                # print(most_common_value)
                max_filtered[y, x] = most_common_value
            else:
                max_filtered[y, x] = input_array[y,x]

    return max_filtered


def clean_image(input_image_path, output_image_path, radius):
    # Load the input image
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)

    thresholded = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    edges = cv2.Canny(thresholded, threshold1=30, threshold2=70)

    # Apply median blur to reduce noise
    blurred_image = cv2.medianBlur(gray_image, 5)

    # Create a structuring element for morphological operations
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)

    # Dilate the image to count pixels within the radius
    dilated_image = cv2.dilate(blurred_image, kernel)
    
    kernel = np.ones((40, 40), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    
    # Find the mode (most frequent pixel value) within the radius
    mode_image = manual_maximum_filter(gray_image, edges_dilated,size=2 * radius + 1)
    
    # Replace each pixel with its corresponding mode
    # cleaned_image = mode_image - dilated_image + blurred_image
    # ax.imshow(mode_image, cmap=color_map)
    # plt.show()
    # Save the cleaned image
    cv2.imwrite(output_image_path, mode_image)

if __name__ == "__main__":
    for file in os.listdir('./from_here'):
        input_image_path = os.path.join('./from_here',file)  # Change this to your input image path
        output_image_path = os.path.join('./from_here_out',file)  # Change this to your desired output image path
        radius = 20  # Adjust the radius as needed

        clean_image(input_image_path, output_image_path, radius)
        print("Image cleaning complete.")
