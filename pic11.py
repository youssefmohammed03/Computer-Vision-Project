from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import binary_erosion
import cv2
from rotation import *

def detect_noise_sine_waves(image, z_score_threshold=2.5, low_freq_cutoff_ratio=0.02, high_freq_cutoff_ratio=0.5):
    """
    Detect noise sine waves in an image while avoiding structured patterns (e.g., barcodes).

    Parameters:
        image (numpy.ndarray): Input image (grayscale).
        z_score_threshold (float): Z-score threshold for detecting noise.
        low_freq_cutoff_ratio (float): Low frequency cutoff as a ratio of image size.
        high_freq_cutoff_ratio (float): High frequency cutoff as a ratio of image size.

    Returns:
        bool: True if noise sine waves are detected, False otherwise.
    """
    # Convert the image to grayscale if it is not
    gray_image = image

    # Step 1: Apply FFT to transform the image to the frequency domain
    dft = np.fft.fft2(gray_image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)

    # Step 2: Compute z-scores of the magnitude spectrum
    mean_mag = np.mean(magnitude_spectrum)
    std_mag = np.std(magnitude_spectrum)
    z_scores = (magnitude_spectrum - mean_mag) / std_mag

    # Step 3: Create a frequency filter
    rows, cols = gray_image.shape
    center_row, center_col = rows // 2, cols // 2
    low_freq_cutoff = int(min(rows, cols) * low_freq_cutoff_ratio)
    high_freq_cutoff = int(min(rows, cols) * high_freq_cutoff_ratio)

    # Create a circular band-pass filter
    y, x = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((x - center_col)*2 + (y - center_row)*2)
    band_pass_filter = (distance_from_center >= low_freq_cutoff) & (distance_from_center <= high_freq_cutoff)

    # Exclude central frequencies (structured patterns like barcodes)
    exclusion_width = int(min(rows, cols) * 0.05)  # Exclude a small width around central axes
    band_pass_filter[center_row - exclusion_width:center_row + exclusion_width, :] = False
    band_pass_filter[:, center_col - exclusion_width:center_col + exclusion_width] = False

    # Apply the band-pass filter to the z-score mask
    sine_wave_mask = (z_scores > z_score_threshold) & band_pass_filter

    # Step 4: Enhance the detected regions using dilation
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size to control dilation
    enhanced_mask = cv2.dilate(sine_wave_mask.astype(np.uint8), kernel, iterations=1)

    # Check if any sine waves were detected
    if np.any(enhanced_mask):
        print("Noise sine waves detected.")
        return True
    else:
        print("No noise sine waves detected.")
        return False

def process_sinewave_image(img_path, filter_direction='horizontal', filter_size=1, filter_width=None, intensity_low=30, intensity_high=210, erosion_size=1, dilation_size=2):
    """
    Process an image by applying a band-stop filter in the frequency domain, modifying intensity, 
    and performing morphological operations to reduce noise.

    Parameters:
    - img_path: str, path to the image file.
    - filter_direction: str, direction of the band-stop filter ('horizontal' or 'vertical').
    - filter_size: int, size of the filter band.
    - filter_width: int or None, width of the filter band. If None, uses full dimension.
    - intensity_low: int, lower bound for zeroing intensities.
    - intensity_high: int, upper bound for zeroing intensities.
    - erosion_size: int, size of the erosion kernel.
    - dilation_size: int, size of the dilation kernel.

    Returns:
    - None, displays the processed image and saves it to a file.
    """
    image = Image.open(img_path).convert('L')  # Convert to grayscale
    f_transform_centered, magnitude_spectrum = transform_image_to_frequency_domain(image)

    # Apply the band-stop filter
    f_transform_filtered = apply_band_stop_filter(f_transform_centered, direction=filter_direction,
                                                  size=filter_size, width=filter_width)
    img_back = fft.ifft2(fft.ifftshift(f_transform_filtered))
    img_inverted = 255 - np.abs(img_back)

    # Zero certain intensity ranges
    img_modified = zero_intensity_range(img_inverted, low=intensity_low, high=intensity_high)

    # Morphological operations: erosion followed by dilation
    mask = cv2.inRange(img_modified, 0, 150)
    kernel = np.ones((erosion_size, erosion_size), np.uint8)  
    eroded_image = cv2.erode(mask, kernel, iterations=1)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    # dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    img_final = 255 - eroded_image

    # Save and plot the final image
    output_path = 'output/11.jpg'
    cv2.imwrite(output_path, img_final)
   

def transform_image_to_frequency_domain(image):
    img_array = np.array(image)
    f_transform = fft.fft2(img_array)
    f_transform_centered = fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_centered) + 1)
    return f_transform_centered, magnitude_spectrum

def apply_band_stop_filter(f_transform, direction='horizontal', size=1, width=None):
    rows, cols = f_transform.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=bool)
    if direction == 'horizontal':
        if width is None:
            width = cols
        start_col = max(0, ccol - width // 2)
        end_col = min(cols, ccol + width // 2)
        mask[crow-size:crow+size, start_col:end_col] = False
    elif direction == 'vertical':
        if width is None:
            width = rows
        start_row = max(0, crow - width // 2)
        end_row = min(rows, crow + width // 2)
        mask[start_row:end_row, ccol-size:ccol+size] = False
    return f_transform * mask

def zero_intensity_range(image, low=30, high=210):
    modified_image = np.copy(image)
    modified_image[(modified_image >= low) & (modified_image <= high)] = 0
    return modified_image


