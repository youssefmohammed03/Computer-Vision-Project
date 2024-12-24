import cv2
import numpy as np

def preprocess_image(image):
    """Convert the image to grayscale."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    """Apply binary thresholding to the grayscale image."""
    # Since the background is white, use THRESH_BINARY_INV to make barcode lines white
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def find_orientation(thresh):
    """Detect edges using Canny edge detector."""
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    """
    Find the angle of the barcode.

    This function uses the Hough Line Transform to detect lines and computes their angle.
    """
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    angles = []

    if lines is None:
        return 0  # Assume no rotation if no lines are detected.

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    if len(angles) == 0:
        return 0

    # Compute the median angle
    median_angle = np.median(angles)
    return median_angle

def needs_rotation(angle):
    """Check if the barcode image needs rotation."""
    # Convert to grayscale
    # Check if the angle is not close to zero
    is_rotated = 90 - abs(angle) > 1
    print(f"Is rotated: {is_rotated}")
    return is_rotated

def rotate_image(image, angle):
    """Rotate the image by the specified angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, -(90-angle), 1.0)

    # Compute sine and cosine (for bounding box calculation)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    # Compute new bounding dimensions
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
    return rotated

def rotate_barcode(image, output_path=None):
    """Preprocess the barcode image and optionally save the corrected image."""

    # Convert to grayscale
    thresh = preprocess_image(image)
    print("Preprocessed image for rotation.")

    # Find orientation
    angle = find_orientation(thresh)
    print(f"Detected angle: {angle:.2f} degrees")

    # Rotate image to make barcode horizontal
    if needs_rotation(angle):
        rotated = rotate_image(image, angle)
        print("Image rotated to correct orientation.")
    else:
        rotated = image
        print("Image is already in correct orientation.")

    # Optionally save the corrected image
    if output_path:
        cv2.imwrite(output_path, rotated)
        print(f"Corrected image saved to {output_path}.")

    return rotated

