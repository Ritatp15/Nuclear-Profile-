# -*- coding: utf-8 -*-
"""
Created on Fri May 30 15:11:57 2025

@author: paulo
"""

# IMPORT UTILS
import os, sys, nd2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage as sk
from skimage import io, filters
from skimage import feature as sk_feature
from skimage import measure as sk_measure
from skimage import segmentation as sk_segmentation
from skimage import morphology as sk_morphology
from skimage.draw import polygon, disk
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt, binary_fill_holes

# --- Helper Functions ---

def saveFig(name, folder='output/'):
    """
    Saves the current matplotlib figure.

    Args:
        name (str): The filename for the saved figure.
        folder (str, optional): The directory to save the figure. Defaults to 'output/'.
    """
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, name), dpi=300, bbox_inches='tight', transparent=False)
    
def readTIF(file_path):
    """
    Reads a TIF file with dimensions (Z, Y, X, C) and performs
    max intensity projection along the Z-axis for all channels.

    Args:
        file_path (str): The path to the TIF file.

    Returns:
        tuple: (c1, c2, c3) where c1, c2, c3 are 2D numpy arrays
               representing the max intensity projections of channels 0, 1, and 2,
               respectively. Returns (None, None, None) on error.
    """
    try:
        tif = io.imread(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None, None, None
    except Exception as e:
        print(f"Error reading TIF file '{file_path}': {e}")
        return None, None, None

    if tif.ndim != 4:
        print(f"Warning: Expected a 4D TIF file (Z, Y, X, C), but got {tif.ndim} dimensions.")
        # Attempt projection on the first dimension if not 4D, assuming it might be Z
        return np.max(tif, axis=0), None, None
    else:
        c1 = np.max(tif[:, :, :, 0], axis=0)  # dapi
        c2 = np.max(tif[:, :, :, 1], axis=0)  # lamina
        c3 = np.max(tif[:, :, :, 2], axis=0)  # irrelevant
        return c1, c2, c3
    
# SEGMENTATION FUNCTIONS

def make_binary(im_raw, sigma=11, closing_radius=1, min_object_size=50, max_hole_size=1000):
    """
    Generates a cleaned binary image from a raw image.

    Args:
        im_raw (numpy.ndarray): The input grayscale image.
        sigma (float, optional): Standard deviation for the Gaussian filter. Defaults to 11.
        closing_radius (int, optional): Radius of the disk structuring element for closing. Defaults to 1.
        min_object_size (int, optional): Minimum size (in pixels) for objects to be kept. Defaults to 50.
        max_hole_size (int, optional): Maximum size (in pixels) for holes to be filled. Defaults to 10000.

    Returns:
        numpy.ndarray: The cleaned binary image.
    """
    if im_raw is None or im_raw.ndim != 2:
        print("Error: Input 'im_raw' must be a 2D numpy array.")
        return None

    # 1. Denoising
    denoised_image = filters.gaussian(im_raw, sigma=sigma)

    # 2. Thresholding
    thresh = filters.threshold_otsu(denoised_image)
    binary = denoised_image > thresh

    # 3. Remove small objects
    cleaned_binary = sk_morphology.remove_small_objects(binary, min_size=min_object_size)

    # 4. Fill smaller holes in the binary mask
    filled_binary = sk_morphology.remove_small_holes(cleaned_binary, area_threshold=max_hole_size)

    # 5. Apply closing operation
    if closing_radius > 0:
        selem = sk_morphology.disk(closing_radius)
        closed_binary = sk_morphology.closing(filled_binary, selem)
        return closed_binary
    else:
        return filled_binary

def watershed_split(im_binary, min_distance=30, clear_border=True, min_segment_size=100):
    """
    Applies watershed segmentation to a binary image to separate objects.

    Args:
        im_binary (numpy.ndarray): The input binary image.
        min_distance (int, optional): Minimum distance between local maxima. Higher values can prevent oversegmentation. Defaults to 30.
        clear_border (bool, optional): Whether to remove segmented objects touching the image border. Defaults to True.
        min_segment_size (int, optional): Minimum size (in pixels) for segmented objects to be kept. Defaults to 100.

    Returns:
        numpy.ndarray: A label map where each segmented object has a unique integer label.
    """
    if im_binary is None or im_binary.ndim != 2 or im_binary.dtype != bool:
        print("Error: Input 'im_binary' must be a 2D boolean numpy array.")
        return None

    # 1. Compute distance transform
    distance = distance_transform_edt(im_binary)

    # 2. Find local maxima of the distance map
    local_max_coords = sk_feature.peak_local_max(distance, min_distance=min_distance, exclude_border=False)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True

    # 3. Label connected regions of local maxima to serve as markers
    markers = sk_measure.label(local_max_mask, connectivity=1)

    # 4. Apply watershed segmentation
    watershed_labels = sk_segmentation.watershed(-distance, markers, mask=im_binary)

    # 5. Remove segmented objects touching the border
    if clear_border:
        watershed_labels = sk_segmentation.clear_border(watershed_labels, buffer_size=1)

    # 6. Remove small segmented objects
    watershed_labels = sk_morphology.remove_small_objects(watershed_labels, min_size=min_segment_size)

    return watershed_labels

def find_contours_from_markers(labeled_image, level=0.5):
    """
    Extracts individual contours for each distinct marker in a labeled image.

    This function assumes the input 'labeled_image' has already undergone
    connected component analysis, where each unique integer value (other than 0)
    represents a distinct object/marker.

    Args:
        labeled_image (numpy.ndarray): A 2D NumPy array (image) where each
                                       connected component (marker) is assigned
                                       a unique integer label (e.g., from
                                       skimage.measure.label or cv2.connectedComponents).
                                       Background should be 0.
        level (float, optional): The iso-value at which to find contours for
                                 each marker. For binary masks (0s and 1s),
                                 0.5 is typically the correct choice.
                                 Defaults to 0.5.

    Returns:
        list: A list of NumPy arrays, where each array represents the
              (row, column) coordinates of a single contour. Each contour
              corresponds to a unique marker in the input image. Returns an
              empty list if no markers are found.
    """
    if not isinstance(labeled_image, np.ndarray) or labeled_image.ndim != 2:
        raise ValueError("Input 'labeled_image' must be a 2D NumPy array.")
    if labeled_image.max() == 0:
        return [] # No markers found if max value is 0 (only background)

    all_individual_contours = []

    # Get the unique labels present in the image, excluding the background (0)
    # np.unique is robust even if labels are not sequential (e.g., 1, 3, 5)
    unique_labels = np.unique(labeled_image)
    foreground_labels = unique_labels[unique_labels != 0]

    for label in foreground_labels:
        # Create a boolean mask for the current marker
        # This is more memory-efficient than converting to uint8 unless strictly needed
        marker_mask = (labeled_image == label)

        # Find contours for this isolated marker's mask
        # skimage.measure.find_contours returns a list of contours (e.g., one for outer, others for holes)
        current_marker_contours = sk_measure.find_contours(marker_mask, level)

        if current_marker_contours:
            # If multiple contours are found for a single marker (e.g., due to holes),
            # we typically want the main outer boundary.
            # Using max(..., key=len) reliably picks the longest contour (outer boundary).
            main_contour = max(current_marker_contours, key=len)
            all_individual_contours.append(main_contour)

    return all_individual_contours

# FUNCTIONS TO GET NUCLEAR PROPERTIES

def get_region_props(markers, im_raw, level = 0.5):
    # Use int32 to avoid overflow for labels >127
    markers = markers.astype(np.int32)
    
    if len(markers.shape) == 3: 
        markers = np.max(markers, axis =2)    
    
    # Relabel to ensure consecutive labels
    from skimage import measure
    markers = measure.label(markers > 0, connectivity=1)
    
    # Get all regions without filtering
    props = measure.regionprops_table(
        markers,
        intensity_image=im_raw,
        properties=[
            'label',
            'area',
            'weighted_centroid',
            'major_axis_length',
            'minor_axis_length',
            'mean_intensity',
            'coords'
        ]
    )
    
    # remove small markers
    #props = props[props['area'] != 1].reset_index(drop=True)

    # add coords of contours to table
    contours = find_contours_from_markers(markers, level)
    props['boundaries'] = contours
    
    # # redefine labels; start with ONE
    # props.label = props.index + 1
    props = pd.DataFrame(props)
    return props


def add_labels_to_fig(ax, regionprops_table, X = "weighted_centroid-1", Y = "weighted_centroid-0", s = 2, color = 'red'):
    '''Adds labels to an existing Matplotlib figure based on axis selection.
    s: size of text; color: text color'''
    
    table = regionprops_table

    for i in range(table.shape[0]):
        label = str(table.iloc[i]["label"].astype(int))
        x = table.iloc[i][X].astype(np.float32)
        y = table.iloc[i][Y].astype(np.float32)
        
        # Add text label to the plot
        ax.text(x, y, label, fontsize=s, color=color, weight='semibold', ha='center', va='center')
    
    return ax
   
def draw_marker_contours(contours, ax, color='red', lw=1):
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=lw)
        
    return contours

# FUNCTIONS TO COMPUTE EFC RATIO

def calculate_single_object_efc_ratio(xy_array):
    """
    Compute the Elliptical Fourier Descriptor (EFC) ratio
    for a single array of XY boundary coordinates.

    Args:
        xy_array (np.ndarray): A 2D NumPy array where each row is an [X, Y] coordinate.

    Returns:
        float: The calculated EFC ratio. Returns np.inf if the denominator is zero.
               Returns np.nan for invalid inputs (e.g., too few points).
    """
    # Basic input validation for empty or invalid arrays
    if not isinstance(xy_array, np.ndarray) or xy_array.ndim != 2 or xy_array.shape[1] != 2:
        # print(f"Warning: Invalid input type or shape for EFC calculation. Expected 2D array with 2 columns. Got: {type(xy_array)} with shape {getattr(xy_array, 'shape', 'N/A')}. Returning NaN.")
        return np.nan
    if xy_array.shape[0] < 3: # Need at least 3 points to form a non-degenerate shape
        # print(f"Warning: Too few points ({xy_array.shape[0]}) in object for EFC calculation. Need at least 3. Returning NaN.")
        return np.nan

    x = xy_array[:, 0]  # X-coordinates
    y = xy_array[:, 1]  # Y-coordinates

    # Ensure the shape is closed by appending the first point to the end
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # Step 1: Compute the distances between consecutive points
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    # Step 2: Compute the cumulative arc length
    arc_length = np.concatenate(([0], np.cumsum(distances)))

    # Step 3: Define the desired number of evenly spaced points
    num_interpolated_points = len(x)

    # Step 4: Create a new set of evenly spaced arc lengths
    # Handle cases where arc_length might have issues (e.g., all points identical)
    if arc_length[-1] == 0: # If total arc length is zero, it's a single point or all identical points
        return np.nan # Cannot calculate EFC for this

    new_arc_length = np.linspace(0, arc_length[-1], num_interpolated_points)

    # Step 5: Interpolate the X and Y coordinates based on the new arc lengths
    # interp1d requires at least 2 points in arc_length to create an interpolator
    if len(arc_length) < 2:
        return np.nan

    interp_x = interp1d(arc_length, x, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(arc_length, y, kind='linear', fill_value="extrapolate")

    x_interp = interp_x(new_arc_length)
    y_interp = interp_y(new_arc_length)

    # Number of points for DFT
    n_points = num_interpolated_points
    t = np.linspace(0, 2 * np.pi, n_points)

    # Perform Discrete Fourier Transform (DFT) on the X and Y coordinates
    cX = np.fft.fft(x_interp) / n_points
    cY = np.fft.fft(y_interp) / n_points

    # Extract the first 20 harmonics
    num_harmonics = 20
    # Python indexing is 0-based. Harmonics 1 to num_harmonics correspond to
    # array indices 1 to num_harmonics in the FFT output (ignoring 0th harmonic).
    harmonics_indices = np.arange(1, num_harmonics + 1)

    # Initialize major and minor axes storage
    major_axes = np.zeros(num_harmonics)
    minor_axes = np.zeros(num_harmonics)

    # Calculate major and minor axes for each harmonic
    for n_idx, n_val in enumerate(harmonics_indices):
        # Ensure n_val is within bounds of cX/cY
        if n_val >= len(cX):
            # This can happen if num_interpolated_points is less than num_harmonics + 1
            break
        a_n = np.abs(cX[n_val])
        b_n = np.abs(cY[n_val])

        major_axes[n_idx] = a_n
        minor_axes[n_idx] = b_n

    # Calculate the EFC ratio
    EFC_numerator = major_axes[0] + minor_axes[0]
    EFC_denominator = np.sum(major_axes[1:] + minor_axes[1:])

    if EFC_denominator != 0:
        EFC_ratio = EFC_numerator / EFC_denominator
    else:
        EFC_ratio = np.inf # Handle division by zero or cases with no higher harmonics

    return EFC_ratio

def add_efc_ratio_to_dataframe(df, coords_column_name = 'boundaries', new_column_name = 'efc_ratio'):
    """
    Applies Elliptical Fourier Descriptor (EFC) ratio calculation to a DataFrame.

    It expects a column containing object boundary coordinates (NumPy arrays).
    A new column with the calculated EFC ratios will be added to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        coords_column_name (str): The name of the column containing the XY coordinates.
                                  Each cell in this column should be a np.ndarray
                                  of shape (N, 2).
        new_column_name (str): The name for the new column to be added,
                               containing the calculated EFC ratios.

    Returns:
        pd.DataFrame: The DataFrame with the new EFC ratio column added.
    """
    if coords_column_name not in df.columns:
        raise ValueError(f"Column '{coords_column_name}' not found in the DataFrame.")

    # Apply the single object EFC ratio function to each row in the specified column
    df[new_column_name] = df[coords_column_name].apply(calculate_single_object_efc_ratio)

    return df

# FUNCTIONS TO COMPUTE RADIAL PROFILE

def calculate_radial_profile(image, mask, n_rings):
    """
    Calculates the total intensity per radial ring for each object in a mask.
    The rings follow the shape of the object's boundary, based on the
    Euclidean Distance Transform (EDT).
    The distribution is measured from the object's boundary (outermost ring)
    to its center (innermost ring).
    The numbering of rings is from 1 (outermost) to N (innermost).

    Args:
        image (np.ndarray): The original intensity image (will be converted to float if not already).
        mask (np.ndarray): A labeled mask where each object has a unique integer label
                           (e.g., from sk_measure.label).
        n_rings (int): The number of equally spaced radial rings to create.

    Returns:
        pandas.DataFrame: A DataFrame where each row represents an object,
        and columns are 'label', 'total_intensity_ring_1', ..., 'total_intensity_ring_N',
        and 'shell_ratio'.
    """

    # Ensure mask is integer labeled (if not already) and boolean for EDT
    if mask.dtype != int:
        mask = sk_measure.label(mask > 0)

    # Ensure image is float for calculations
    if image.dtype != float:
        image = image.astype(np.float32)

    radial_profiles = []

    for region in sk_measure.regionprops(mask, image):
        label = region.label
        if label == 0: # Skip background label
            continue

        object_mask = (mask == label) # Boolean mask for the current object

        # Calculate the Euclidean Distance Transform (EDT) for the current object
        distance_map = distance_transform_edt(object_mask)

        # Get the maximum distance within the object, which defines its "radius" along its longest axis
        max_distance = np.max(distance_map)

        # Handle single pixel objects or objects with no measurable extent
        if max_distance == 0:
            row_data = {'label': label}
            # Assign the sum of intensity for a single pixel to all rings, or NaN if no intensity

            for i in range(n_rings):
                row_data[f'ring{i + 1}_total_intensity'] = region.intensity_image.sum() if region.intensity_image.size > 0 else np.nan
            row_data['shell_ratio'] = np.nan # Shell ratio is not meaningful for single pixel
            radial_profiles.append(row_data)
            continue # Move to the next object

        # Create equally spaced distance intervals (ring boundaries)
        # These boundaries define the concentric rings from 0 (boundary) to max_distance (center)
        ring_boundaries = np.linspace(0, max_distance, n_rings + 1)

        ring_total_intensities = [np.nan] * n_rings # Initialize with NaN for empty rings

        # Iterate through the rings to calculate intensity
        # The 'calculate_radial_profile' function defines rings from outermost (1) to innermost (N).
        # This implies:
        # Ring 1: distance_map values between ring_boundaries[0] and ring_boundaries[1] (outermost)
        # Ring N: distance_map values between ring_boundaries[N-1] and ring_boundaries[N] (innermost)

        for i in range(n_rings):
            lower_bound = ring_boundaries[i]
            upper_bound = ring_boundaries[i + 1]

            # Identify pixels within the current ring
            # A pixel belongs to a ring if its distance transform value falls within the ring's bounds
            # and it is part of the current object.
            ring_pixels_mask = (distance_map >= lower_bound) & (distance_map < upper_bound) & object_mask

            if np.any(ring_pixels_mask):
                ring_intensities_values = image[ring_pixels_mask]
                total_intensity = np.sum(ring_intensities_values)
                ring_total_intensities[i] = total_intensity # Store in correct ring index

            # Else, it remains NaN as initialized

        # Prepare row data for DataFrame
        row_data = {'label': label}
        for i, total_intensity in enumerate(ring_total_intensities):
            row_data[f'ring{i + 1}_total_intensity'] = total_intensity

        # Calculate shell_ratio
        shell_ratio = np.nan
        if n_rings >= 1:
            outermost_ring_intensity = ring_total_intensities[0] # Ring 1 is at index 0

            # Get intensities of all other rings (from ring 2 to N)
            other_rings_intensities = [val for val in ring_total_intensities[1:] if not np.isnan(val)]

            if len(other_rings_intensities) > 0:
                average_other_rings = np.mean(other_rings_intensities)
                if average_other_rings != 0:
                    shell_ratio = outermost_ring_intensity / average_other_rings
                else:
                    shell_ratio = np.inf # Handle division by zero if average of other rings is zero
            elif n_rings == 1:
                # If only one ring, there are no "other rings", so ratio is undefined
                shell_ratio = np.nan
            else:
                # If other_rings_intensities is empty (e.g., all were NaN)
                shell_ratio = np.nan
        row_data['shell_ratio'] = shell_ratio
        radial_profiles.append(row_data)

    # Convert to DataFrame
    df = pd.DataFrame(radial_profiles)

    return df

# FUNCTIONS TO VISUALIZE EFC OBJECTS

def get_efc_components(xy_array, num_harmonics_for_ratio=20):
    """
    Computes Elliptical Fourier Descriptor components for a single object's boundary.
    Returns the EFC ratio, Fourier coefficients, interpolated coordinates, and time vector.

    Args:
        xy_array (np.ndarray): A 2D NumPy array where each row is an [X, Y] coordinate.
        num_harmonics_for_ratio (int): The number of harmonics to use for EFC ratio calculation.

    Returns:
        tuple: (efc_ratio, cX, cY, x_interp, y_interp, t)
               Returns NaNs/None if inputs are invalid.
    """
    if not isinstance(xy_array, np.ndarray) or xy_array.ndim != 2 or xy_array.shape[1] != 2:
        return np.nan, None, None, None, None, None
    if xy_array.shape[0] < 3:
        return np.nan, None, None, None, None, None

    x = xy_array[:, 0]
    y = xy_array[:, 1]

    x = np.append(x, x[0])
    y = np.append(y, y[0])

    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    arc_length = np.concatenate(([0], np.cumsum(distances)))

    if arc_length[-1] == 0:
        return np.nan, None, None, None, None, None

    num_interpolated_points = len(x)
    new_arc_length = np.linspace(0, arc_length[-1], num_interpolated_points)

    if len(arc_length) < 2:
        return np.nan, None, None, None, None, None

    interp_x = interp1d(arc_length, x, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(arc_length, y, kind='linear', fill_value="extrapolate")

    x_interp = interp_x(new_arc_length)
    y_interp = interp_y(new_arc_length)

    n_points = num_interpolated_points
    t = np.linspace(0, 2 * np.pi, n_points)

    cX = np.fft.fft(x_interp) / n_points
    cY = np.fft.fft(y_interp) / n_points

    major_axes = np.zeros(num_harmonics_for_ratio)
    minor_axes = np.zeros(num_harmonics_for_ratio)
    harmonics_indices = np.arange(1, num_harmonics_for_ratio + 1)

    for n_idx, n_val in enumerate(harmonics_indices):
        if n_val >= len(cX): break
        a_n = np.abs(cX[n_val])
        b_n = np.abs(cY[n_val])
        major_axes[n_idx] = a_n
        minor_axes[n_idx] = b_n

    EFC_numerator = major_axes[0] + minor_axes[0]
    EFC_denominator = np.sum(major_axes[1:] + minor_axes[1:])
    efc_ratio = EFC_numerator / EFC_denominator if EFC_denominator != 0 else np.inf

    return efc_ratio, cX, cY, x_interp, y_interp, t

# --- Main Plotting Function: Overlays all on one plot (accepts optional ax) ---

def display_object_efc_analysis_overlay(
    xy_array: np.ndarray,
    object_id: int = None,
    binary_mask: np.ndarray = None,
    image_shape: tuple = None, # (height, width) of the original full image
    harmonics_to_plot_list: list = None,
    title_prefix: str = "Object Analysis",
    efc_num_harmonics: int = 20, # Number of harmonics to use for the EFC ratio calculation
    cmap_name: str = 'cool', # Colormap for harmonic lines
    ax: plt.Axes = None # <--- NOW OPTIONAL: The Axes object to plot on
):
    """
    Displays an object's binary mask (if provided), its raw contour,
    and harmonic reconstructions all on the same plot, with labels and colors.
    If 'ax' is not provided, a new figure and axes will be created and displayed.

    Args:
        xy_array (np.ndarray): 2D array of [X, Y] boundary coordinates for the object.
                               Expected to be in (column, row) format for plotting against image.
        object_id (int, optional): An identifier for the object, used in titles.
        binary_mask (np.ndarray, optional): A 2D boolean or 0/1 array representing the
                                             isolated binary mask of the object. If provided,
                                             it will be imshow'd as the background.
        image_shape (tuple, optional): (height, width) of the original full image.
                                        Used for setting plot limits if no binary_mask is given.
        harmonics_to_plot_list (list, optional): List of integers for the number of
                                                  harmonics to plot for reconstruction.
                                                  Defaults to [1, 5, 10, 20].
        title_prefix (str, optional): A prefix for the main plot title.
        efc_num_harmonics (int, optional): Number of harmonics to use for the EFC ratio calculation.
        cmap_name (str): Name of the Matplotlib colormap to use for different harmonic lines.
        ax (plt.Axes, optional): The Matplotlib Axes object on which to draw the plots.
                                 If None, a new figure and axes will be created.
    """
    _display_and_close_figure = False
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10)) # Create new figure and axes
        _display_and_close_figure = True
    else:
        # If an axis is provided, clear it before plotting new content
        ax.clear()

    if harmonics_to_plot_list is None:
        harmonics_to_plot_list = [1, 5, 10, 20]

    efc_ratio, cX, cY, x_interp, y_interp, t = get_efc_components(xy_array, efc_num_harmonics)

    if cX is None:
        print(f"Error: Could not process xy_array for object ID {object_id}. Skipping plot.")
        if _display_and_close_figure:
            plt.close(fig) # Close the newly created figure if an error occurred
        return

    x_min, x_max = x_interp.min(), x_interp.max()
    y_min, y_max = y_interp.min(), y_interp.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    padding_x = x_range * 0.1 if x_range > 0 else 10
    padding_y = y_range * 0.1 if y_range > 0 else 10
    plot_xlim = [x_min - padding_x, x_max + padding_x]
    plot_ylim = [y_min - padding_y, y_max + padding_y]

    if image_shape:
        plot_xlim = [0, image_shape[1]]
        plot_ylim = [image_shape[0], 0]

    main_title = (f"{title_prefix} (ID: {object_id if object_id is not None else 'N/A'})\n"
                  f"EFC Ratio: {efc_ratio:.3f}")
    ax.set_title(main_title, fontsize=8)

    if binary_mask is not None:
        mask_height, mask_width = binary_mask.shape
        ax.imshow(binary_mask, cmap='gray', origin='upper',
                  extent=[0, mask_width, mask_height, 0], alpha=0.5)
        ax.set_xlim(0, mask_width)
        ax.set_ylim(mask_height, 0)
    else:
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)

    ax.plot(x_interp, y_interp, 'k-', linewidth=1.5, label='Raw Contour')

    colors = plt.cm.get_cmap(cmap_name, len(harmonics_to_plot_list))

    for i, num_h in enumerate(harmonics_to_plot_list):
        if num_h >= len(cX):
            warnings.warn(f"Cannot reconstruct with {num_h} harmonics. Max available is {len(cX) - 1}. Skipping harmonic {num_h}.")
            continue

        x_reconstructed = np.zeros_like(t, dtype=float)
        y_reconstructed = np.zeros_like(t, dtype=float)

        x_reconstructed += np.real(cX[0] * np.exp(1j * 0 * t))
        y_reconstructed += np.real(cY[0] * np.exp(1j * 0 * t))

        for n_val in range(1, num_h + 1):
            if n_val >= len(cX):
                break
            x_reconstructed += np.real(cX[n_val] * np.exp(1j * n_val * t))
            y_reconstructed += np.real(cY[n_val] * np.exp(1j * n_val * t))

        ax.plot(x_reconstructed, y_reconstructed, color=colors(i), linestyle='-',
                linewidth=1.5, label=f'{num_h} Harmonics')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', frameon=False, framealpha=0.8, fontsize=7)

    # Only call tight_layout and show if the function created the figure
    if _display_and_close_figure:
        plt.tight_layout()
        plt.show()
        
# FUNCTIONS TO VISUALIZA RADIAL RINGS

def get_radial_ring_outlines(mask: np.ndarray, n_rings: int = 5):
    """
    Computes the contour coordinates for each radial ring (band) for every object in a mask.
    The rings follow the shape of the object's boundary, based on the Euclidean Distance Transform (EDT).
    Each ring is defined as a band of pixels at a specific distance range from the object's boundary,
    consistent with how `calculate_radial_profile` defines its intensity summation regions.
    The rings are numbered from 1 (outermost) to N (innermost).

    Args:
        mask (np.ndarray): A labeled mask where each object has a unique integer label.
                           Expected to be int dtype.
        n_rings (int): The number of equally spaced radial rings (bands) to define.

    Returns:
        dict: A dictionary where keys are object labels (int) and values are lists of lists of NumPy arrays.
              Each inner list corresponds to a ring (band), and contains one or more NumPy arrays,
              where each array represents a contour of that ring's boundary (row, column) coordinates.
              The outer list of rings for each object is ordered from outermost (Ring 1) to innermost (Ring N).
              Example:
              {
                  1: [
                      [array_contour_obj1_ring1_part1, array_contour_obj1_ring1_part2], # Ring 1 (outermost)
                      [array_contour_obj1_ring2_part1],                                # Ring 2
                      ...
                      [array_contour_obj1_ringN_part1]                                 # Ring N (innermost)
                  ],
                  2: [ ... ],
                  ...
              }
    """
    if mask.dtype != int:
        mask = sk_measure.label(mask > 0)

    all_objects_ring_outlines = {}

    for region in sk_measure.regionprops(mask):
        label = region.label
        if label == 0: # Skip background label
            continue

        object_mask = (mask == label) # Boolean mask for the current object within its bbox
        distance_map = distance_transform_edt(object_mask)
        max_distance = np.max(distance_map)

        # Handle objects too small for meaningful rings
        if max_distance <= 0.5:
            all_objects_ring_outlines[label] = [[] for _ in range(n_rings)] # Return empty lists for each ring
            continue

        # Create equally spaced distance intervals (ring boundaries)
        # These boundaries define the concentric rings from 0 (boundary) to max_distance (center)
        # Example: for n_rings=5, max_distance=10, ring_boundaries would be [0, 2, 4, 6, 8, 10]
        ring_boundaries_distances = np.linspace(0, max_distance, n_rings + 1)

        current_object_rings_contours = []

        # Iterate through the rings to get their pixel masks and then their contours.
        # Ring 1: distance_map values between ring_boundaries_distances[0] and [1] (outermost)
        # Ring N: distance_map values between ring_boundaries_distances[N-1] and [N] (innermost)
        for i in range(n_rings):
            lower_bound_dist = ring_boundaries_distances[i]
            upper_bound_dist = ring_boundaries_distances[i + 1]

            # Identify pixels within the current ring band
            # A pixel belongs to a ring if its distance transform value falls within the ring's bounds
            # and it is part of the current object.
            ring_pixels_mask = (distance_map >= lower_bound_dist) & \
                               (distance_map < upper_bound_dist) & object_mask

            # Find contours for this specific ring's pixel band
            # Note: A single ring band might have multiple disconnected contours if it's complex,
            # or if it surrounds holes. We collect all of them.
            contours_for_this_ring = sk_measure.find_contours(ring_pixels_mask, 0.5)

            current_object_rings_contours.append(contours_for_this_ring[0])

        all_objects_ring_outlines[label] = current_object_rings_contours

    return all_objects_ring_outlines
	
	
# FUNCTIONS TO COUNT SPOTS

def find_local_maxima(im_spots, threshold=0.5, min_dist=10, max_peaks=300,
                      min_area=5, max_area=80, max_eccentricity=0.8):
    
    im_blur = sk.filters.gaussian(im_spots, 1)

    # Step 1: Detect peaks
    coords = sk_feature.peak_local_max(im_blur,
                                       threshold_rel=threshold,
                                       min_distance=min_dist,
                                       num_peaks=max_peaks)

    valid_peaks = []

    for y, x in coords:
        # Step 2: extract a small patch around the peak
        radius = 5
        y_min, y_max = max(0, y - radius), min(im_spots.shape[0], y + radius + 1)
        x_min, x_max = max(0, x - radius), min(im_spots.shape[1], x + radius + 1)
        patch = im_blur[y_min:y_max, x_min:x_max]

        # Step 3: simple segmentation (Otsu or fixed threshold)
        thresh = sk.filters.threshold_otsu(patch)
        binary = patch > thresh
        labeled = sk_measure.label(binary)

        # Step 4: region properties
        regions = sk_measure.regionprops(labeled)

        for region in regions:
            if region.area < min_area or region.area > max_area:
                continue
            if region.eccentricity > max_eccentricity:
                continue
            # Optionally: check if peak is inside region.bbox
            valid_peaks.append([y, x])
            break  # only one region per peak

    return np.array(valid_peaks)

def count_spots_inside_cells(spot_coordinates: np.ndarray,
                             nucleous_properties: pd.DataFrame,
                             spot_count_col_name: str = 'spot_count') -> tuple[pd.DataFrame, np.ndarray]:
    """
    Counts spots inside each cell nucleus, excluding those on the boundary.

    This function determines spot containment using polygonal boundaries of cell
    nuclei and returns both the counts per nucleus and the coordinates of the
    contained spots.

    Parameters
    ----------
    spot_coordinates : np.ndarray
        A NumPy array of shape `(N, 2)` with the (y, x) coordinates of spots.
    nucleous_properties : pd.DataFrame
        A DataFrame with properties of detected nuclei. Must include a 'boundaries'
        column containing NumPy arrays of polygon vertices for each nucleus.
        A 'label' column is highly recommended.
    spot_count_col_name : str, optional
        The name for the output column storing spot counts. Defaults to 'spot_count'.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        A tuple containing:
        - A DataFrame with 'label' and `spot_count_col_name` columns.
        - A NumPy array of shape `(K, 2)` containing the coordinates of the K
          spots that were found inside any nucleus.

    Raises
    ------
    TypeError
        If inputs are not of the expected type (NumPy array, Pandas DataFrame).
    ValueError
        If inputs have incorrect shapes or are missing required columns.
    """

    from matplotlib import path
    
    # --- Input Validation ---
    if not isinstance(spot_coordinates, np.ndarray):
        raise TypeError("`spot_coordinates` must be a NumPy array.")
    if spot_coordinates.ndim != 2 or spot_coordinates.shape[1] != 2:
        raise ValueError("`spot_coordinates` must be a 2D NumPy array of shape `(N, 2)`.")

    if not isinstance(nucleous_properties, pd.DataFrame):
        raise TypeError("`nucleous_properties` must be a Pandas DataFrame.")

    required_cols = ['boundaries']
    if not all(col in nucleous_properties.columns for col in required_cols):
        missing = [col for col in required_cols if col not in nucleous_properties.columns]
        raise ValueError(f"`nucleous_properties` is missing required columns: {missing}.")

    # --- Edge Case Handling ---
    empty_coords = np.array([]).reshape(0, 2)
    if nucleous_properties.empty or nucleous_properties['boundaries'].isnull().all():
        print("Warning: `nucleous_properties` or its 'boundaries' are empty.")
        return pd.DataFrame(columns=['label', spot_count_col_name]), empty_coords

    if 'label' in nucleous_properties.columns:
        all_cell_labels = nucleous_properties['label'].unique()
    else:
        print("Warning: No 'label' column found. Using 1-based index for labels.")
        all_cell_labels = np.arange(1, len(nucleous_properties) + 1)

    if spot_coordinates.size == 0:
        print("Warning: `spot_coordinates` is empty. All spot counts will be zero.")
        counts_df = pd.DataFrame({'label': all_cell_labels, spot_count_col_name: 0})
        return counts_df, empty_coords

    # --- Core Logic ---
    cell_contours = nucleous_properties['boundaries'].tolist()
    cells_with_spots_inside = []
    
    # This boolean mask will track which spots are inside *any* of the cells.
    all_spots_inside_mask = np.zeros(len(spot_coordinates), dtype=bool)

    # Iterate through each cell's contour to check for spot containment.
    for i, cell_contour in enumerate(cell_contours):
        cell_label = nucleous_properties['label'].iloc[i] if 'label' in nucleous_properties.columns else i + 1

        # Create a Path object from the cell's boundary vertices.
        # This is required for the contains_points method.
        cell_path = path.Path(cell_contour)

        # Check which spots are inside the current cell's path.
        # A small negative radius is used to exclude points on the boundary.
        is_spot_inside = cell_path.contains_points(spot_coordinates, radius=25)
        spot_count = np.count_nonzero(is_spot_inside)

        cells_with_spots_inside.append([cell_label, spot_count])
        
        # Update the master mask. A spot is kept if it's in at least one cell.
        all_spots_inside_mask |= is_spot_inside

    # --- Format and Return Output ---
    # Create the DataFrame for spot counts per cell.
    cells_with_spots_inside_df = pd.DataFrame(cells_with_spots_inside, columns=['label', spot_count_col_name])

    # Filter the original coordinates to get only the spots inside nuclei.
    spots_inside_nuclei = spot_coordinates[all_spots_inside_mask]

    return cells_with_spots_inside_df, spots_inside_nuclei
