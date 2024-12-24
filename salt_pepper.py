import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_salt_pepper(image, output_path):
    # Load the image in grayscale
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if image is None:
    #     raise ValueError("Image not found or unable to load.")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Plot the original image
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 5, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # Apply vertical blur to reduce vertical noise streaks
    vertical_blur = cv2.blur(image, (1, 7))

    # Plot the vertically blurred image
    plt.subplot(1, 5, 2)
    plt.title("Vertical Blur")
    plt.imshow(vertical_blur, cmap='gray')
    plt.axis('off')

    # Apply median blur to reduce salt-and-pepper noise further
    median_blur = cv2.medianBlur(vertical_blur, 3)

    # Plot the median blurred image
    plt.subplot(1, 5, 3)
    plt.title("Median Blur")
    plt.imshow(median_blur, cmap='gray')
    plt.axis('off')

    # Apply Otsu's thresholding for binary segmentation
    _, thresholded_image = cv2.threshold(median_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Plot the thresholded image
    plt.subplot(1, 5, 4)
    plt.title("Thresholded Image")
    plt.imshow(thresholded_image, cmap='gray')
    plt.axis('off')

    # Apply vertical median blur to clean remaining noise stuck to bars
    vertical_median = cv2.medianBlur(thresholded_image, 1, 444444445)

    # Plot the vertically median blurred image
    plt.subplot(1, 5, 5)
    plt.title("Vertical Median Blur")
    plt.imshow(vertical_median, cmap='gray')
    plt.axis('off')

    # Save the final processed image
    cv2.imwrite(output_path, vertical_median)

    # Display the plots
    plt.tight_layout()
    plt.show()

    return output_path


