import argparse

import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
# import the necessary libraries
# 1.png --- sample image/original image
# 2.png --- reference image
IMG_PATH = os.path.join('Images','1.png')
img = cv2.imread(IMG_PATH)

# output directory that will save all the images
OUTPUT_DIR = 'output_images'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def main(input_image_path):
# Now we will use the Gaussian Blur filter to reduce the noise and for normalization of color

 img_noise_reduced = cv2.GaussianBlur(img, (5, 5), 0)

 REF_PATH = os.path.join('Images', '2.png')
 reference_img = cv2.imread(REF_PATH)

# normalizing this image
 img_lab = cv2.cvtColor(img_noise_reduced, cv2.COLOR_BGR2Lab)
 ref_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2Lab)


 img_mean, img_std = cv2.meanStdDev(img_lab)
 ref_mean, ref_std = cv2.meanStdDev(ref_lab)


 result_lab = np.zeros(img_lab.shape, dtype=np.uint8)
 for i in range(3):
    result_lab[:, :, i] = ref_std[i] / img_std[i] * (img_lab[:, :, i] - img_mean[i]) + ref_mean[i]


# To apply CLAHE on the normalized image
 img_color_normalized = cv2.cvtColor(result_lab, cv2.COLOR_Lab2BGR)


 img_lab = cv2.cvtColor(img_color_normalized, cv2.COLOR_BGR2Lab)
 clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


 img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])


 img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
 return img_enhanced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and draw circles in an image.")
    parser.add_argument("input_image", type=str, help="Images/1.png")

    args = parser.parse_args()
    main(args.input_image)

# Save the original image
original_img = os.path.join(OUTPUT_DIR, 'Sample Image.png')
cv2.imwrite(original_img, img)

# Save the enhanced image
original_img = os.path.join(OUTPUT_DIR, 'Enhanced Image.png')
cv2.imwrite(original_img, main(IMG_PATH))

# Use Sobel Operator to find the edges

sobel_x = cv2.Sobel(main(IMG_PATH), cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(main(IMG_PATH), cv2.CV_64F, 0, 1, ksize=3)


sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Save the Sobel X, Sobel Y, and Sobel Combined images
output_sobel_x_path = os.path.join(OUTPUT_DIR, 'sobel_x.png')
output_sobel_y_path = os.path.join(OUTPUT_DIR, 'sobel_y.png')
output_sobel_combined_path = os.path.join(OUTPUT_DIR, 'Sobel Filtered Image.png')
cv2.imwrite(output_sobel_x_path, sobel_x)
cv2.imwrite(output_sobel_y_path, sobel_y)
cv2.imwrite(output_sobel_combined_path, sobel_combined)


# Now I will use the lower and upper threshold of the color purple to find edges in color space

# Load the histogram image
histogram_image = main(IMG_PATH)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(histogram_image, cv2.COLOR_BGR2HSV)

# obtain a binary mask of the purple regions
lower_purple = np.array([50, 50,130])
upper_purple = np.array([255, 255,170 ])
mask_purple = cv2.inRange(hsv_image, lower_purple, upper_purple)

# Contour detection
contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out non-circular regions
hepatocytes_image = histogram_image.copy()
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(approx) >= 5:  # To check if the contour is circular
        cv2.drawContours(hepatocytes_image, [contour], -1, (0, 255, 0), 2)


# Save the edges image with potential cells
output_edges_path = os.path.join(OUTPUT_DIR, 'Edges.png')
cv2.imwrite(output_edges_path, hepatocytes_image)


# To identify hypothecites on the "Sobel Combined" and accordingly draw circles on our initial image
# to grayscale
gray = cv2.cvtColor(sobel_combined, cv2.COLOR_BGR2GRAY)

# Blur the image to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect circles using HoughCircles

# We can adjust the threshold values according to any image!
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=40, maxRadius=55)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)


# Save the original image with detected circles
output_detected_circles_path = os.path.join(OUTPUT_DIR, 'Final Output.png')
cv2.imwrite(output_detected_circles_path, img)
cv2.waitKey(0)
cv2.destroyAllWindows()