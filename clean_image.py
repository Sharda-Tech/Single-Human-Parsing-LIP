import cv2
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import argparse

def main():
    
    parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--save_folder', type=str, help='Path to the folder saving images')
    args = parser.parse_args()
    

    for image_filename in os.listdir(args.image_folder):
        print(image_filename)
        if 'person_mask' in image_filename: 
            save_filename = image_filename.split('.')[0] + '_cleaned.png'   
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
            kernel = np.ones((15, 15), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)

            edge_image = np.zeros_like(image, dtype=np.uint8)
            edge_image[edges_dilated > 0] = 255


            masked_image = np.logical_and(gray, edge_image[:,:,0]).astype(np.uint8)
            masked_image = (masked_image * (255.0 / masked_image.max())).astype(np.uint8)

            for row in range(masked_image.shape[0]):
                for col in range(masked_image.shape[1]):
                    if masked_image[row, col] == 255:
                        # masked_image[row, col] = gray[row, col]
                        gray[row, col] = 0
            
            cv2.imwrite(os.path.join(args.save_folder,save_filename),gray)
            
            
if __name__ == '__main__':
    main()

