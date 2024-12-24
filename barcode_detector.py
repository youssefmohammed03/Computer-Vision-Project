import cv2
import numpy as np

import cv2
import numpy as np

def process_barcode_image(image):
    # Load the image
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if image is None:
    #     raise FileNotFoundError("Image file not found. Make sure the path is correct.")

    # Step 1: Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

    # Step 2: Compute gradients
    grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * (180.0 / np.pi)
    direction[direction < 0] += 180

    # Step 3: Non-maximum suppression (optimized)
    nms_image = np.zeros_like(magnitude, dtype=np.uint8)
    rows, cols = magnitude.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = direction[i, j]
            q, r = 255, 255
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                nms_image[i, j] = magnitude[i, j]

    # Step 4: Double thresholding
    low_threshold = 50
    high_threshold = 150
    strong_pixel = 255
    weak_pixel = 75
    strong_edges = (nms_image >= high_threshold)
    weak_edges = ((nms_image >= low_threshold) & (nms_image < high_threshold))
    thresh_image = np.zeros_like(nms_image, dtype=np.uint8)
    thresh_image[strong_edges] = strong_pixel
    thresh_image[weak_edges] = weak_pixel

    # Steps of Contour Analysis
    # Step 1: Find Contours
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Analyze Contours to find the bounding rectangle encompassing the full barcode
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Ensure we have valid coordinates for the bounding box
    if x_min < x_max and y_min < y_max:
        print(f"Bounding Box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

        # Crop the image to the bounding box
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Further crop to remove white borders at the edges of the barcode
        cropped_image = crop_white_space(cropped_image)

        # Display the cropped image using OpenCV
        cv2.imshow("Cropped Barcode Image", cropped_image)

        # Visualize the bounding box on the original image
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color image for visualization
        cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow("Original Image with Bounding Box", output_image)

        # Save the cropped barcode to a file
        cv2.imwrite('Cropped_Barcode.jpg', cropped_image)

        # Wait for a key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return cropped_image
    else:
        print("No valid bounding box found.")
        return None

def crop_white_space(image):
    """
    Crops the white space from the start and end of the image, specifically vertical columns.
    """
    # Remove white columns
    col_sum = np.sum(image, axis=0)
    first_non_white_col = np.argmax(col_sum < 250 * image.shape[0])  # Adjust threshold
    last_non_white_col = image.shape[1] - np.argmax(np.flip(col_sum) < 250 * image.shape[0])
    image = image[:, first_non_white_col:last_non_white_col]

    # Further refine to ensure no remaining white spaces
    while np.all(image[:, 0] >= 250):  # Check if the first column is entirely white
        image = image[:, 1:]
    while np.all(image[:, -1] >= 250):  # Check if the last column is entirely white
        image = image[:, :-1]

    return image
# Example usage:

cropped = process_barcode_image(cv2.imread("test cases/01 easy.jpg", cv2.IMREAD_GRAYSCALE))
