import cv2
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

colormap = [
    (0, 0, 0),
    (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
    (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
    (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
    (1, 0, 0.75), (0, 0.5, 0.5), (0.25, 0.5, 0.5), (1, 0, 0),
    (1, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0)
]

image_filenames = ['00891_00_mask.png', '03615_00_mask.png', '07445_00_mask.png', '07573_00_mask.png', '08909_00_mask.png', '10549_00_mask.png']

color_map= ListedColormap(colormap)

num_rows = len(image_filenames)
fig, ax = plt.subplots(1, 1)

for i, image_filename in enumerate(image_filenames):
    uploaded_image = Image.open(os.path.join('./from_here',image_filename)).convert("L")
    image = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    thresholded = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    edges = cv2.Canny(thresholded, threshold1=30, threshold2=70)
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the outer contours on a copy of the original image
    contour_image = image.copy()
    #cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    for c in contours:
        cv2.drawContours(contour_image, c, -1, (0, 255, 0), 2)
        cv2.imshow('edge',contour_image)
        cv2.waitKey(0)  
    # cv2.imshow('Canny Edges', edges)
    cv2.imshow('edge',contour_image)
    cv2.waitKey(0)

#     kernel = np.ones((15, 15), np.uint8)
#     edges_dilated = cv2.dilate(edges, kernel, iterations=1)

#     edge_image = np.zeros_like(image, dtype=np.uint8)
#     edge_image[edges_dilated > 0] = 255


#     masked_image = np.logical_and(gray, edge_image[:,:,0]).astype(np.uint8)
#     masked_image = (masked_image * (255.0 / masked_image.max())).astype(np.uint8)

#     for row in range(masked_image.shape[0]):
#         for col in range(masked_image.shape[1]):
#             if masked_image[row, col] == 255:
#                 # masked_image[row, col] = gray[row, col]
#                 gray[row, col] = 0
    
#     # color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#     # colored_gradient = cv2.applyColorMap(color_image, colormap)
#     # cv2.imshow('masked_image',colored_gradient)
#     # cv2.waitKey(0)
#     # print("Unique values for image {}: {}".format(i+1, np.unique(masked_image)))
#     # print(masked_image.shape)
#     # print(gray.shape)

#     ax.imshow(gray, cmap=color_map)
#     plt.savefig(fname=os.path.join("./preprocessed",image_filename))
# plt.tight_layout()
# plt.show()
