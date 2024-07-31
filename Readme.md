## HEPATOCYTE CELL DETECTION
This Python code aims to detect hypotocyte cells in an input image using image processing techniques. The code performs the following steps:

# PROCEDURE FOLLOWED
Preprocess the input image by applying Gaussian Blur and color normalization.
Enhance the image using CLAHE (Contrast Limited Adaptive Histogram Equalization).
Detect edges in the image using the Sobel operator.
Identify potential cell regions based on color thresholding and contour detection.
Draw contours around the potential cell regions on the "Edges" image.
Detect circles in the Sobel Combined image using HoughCircles and draw them on the original input image.

# INSTALLATION DEPENDENCIES
1. To install OpenCV library that I have used in my code, you can use this: pip install opencv-python
2. To install the mathplotlib library use: pip install matplotlib
3. To run the python script write this command in the terminal : python histology.py Images/1.png



# Cell-detection-project-
