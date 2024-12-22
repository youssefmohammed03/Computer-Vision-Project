import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_thresholds(img, percentage=0.05):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram
    histogram = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    
    # Calculate the cumulative distribution of the histogram
    cumulative_distribution = np.cumsum(histogram)
    
    # Total number of pixels in the image
    total_pixels = cumulative_distribution[-1]
    
    # Calculate the threshold pixel count for the given percentage
    threshold_pixel_count = total_pixels * percentage
    
    # Find the smallest pixel value whose frequency is at least 5%
    smallest_pixel_value = np.searchsorted(cumulative_distribution, threshold_pixel_count)
    
    # Find the largest pixel value whose frequency is at least 5%
    largest_pixel_value = np.searchsorted(cumulative_distribution, total_pixels - threshold_pixel_count)
    
    return smallest_pixel_value, largest_pixel_value


def contrast_correction(img):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate the average pixel intensity
    avg_intensity = np.mean(img_gray)
    print(avg_intensity)
    '''
    0 -> Black
    255 -> White

    any thing smaller than the threshold is considered black (0)
    any thing larger than the threshold is considered white (255)
    hence the name binary thresholding
    
    '''
    smallest_pixel_value, largest_pixel_value = get_thresholds(img, 0.05)
    img_median = (smallest_pixel_value + largest_pixel_value) // 2

    _, thresholded_image = cv2.threshold(img_gray, img_median, 255, cv2.THRESH_BINARY)
    return thresholded_image
 
# test the function
img = cv2.imread('test cases/09 - e3del el soora ya3ammm.jpg', )
thresholded_image = contrast_correction(img)
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()