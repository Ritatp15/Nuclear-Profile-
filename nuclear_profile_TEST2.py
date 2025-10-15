# -*- coding: utf-8 -*-
"""
Created on Fri May 30 15:13:42 2025

@author: paulo
"""

import sys, os
import argparse
import nd2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Set up Path for Custom Functions --- #
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

# --- Import custom functions --- #
from src.bkg_func import *

# APPLY SEGMENTATION TO AN IMAGE
# COMPUTE INTENSITY FEATURES

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Apply segmentation and compute intensity features on microscopy images.")

    parser.add_argument("input_file", help="Path to the image file (.nd2 or .tif).")

    parser.add_argument(
        "--nucleous_channel", "-nc",
        default='c1',
        help="Channel(s) to consider for nucleus segmentation. Can be 'c1', 'c2', 'c3', or a combination like 'c1+c2', 'c1+c3', 'c2+c3'.")

    parser.add_argument(
        "--intensity_channel", "-ic",
        choices=['c1', 'c2', 'c3'],
        default='c2',
        help="Channel to consider for intensity-based calculations (e.g., radial profile). Options: c1, c2 (default), c3.")

    # Arguments for SPOT Counting	
	
    parser.add_argument(
		"--enable_spot_counting", "-esc",
        action="store_true", # if activated, the argument args.enable_spot_counting is "True"
        help="Enable spot counting analysis and output spot images.")
		
    parser.add_argument(
        "--spot_channel", "-sc",
        choices=['c1', 'c2', 'c3'],
        default='c2', # default channel, but execution depends on the --enable_spot_counting
        help="Channel to consider for intensity-based spot counting. Options: c1, c2 (default), c3.")
		
    parser.add_argument(
        "--spot_min_dist", "-d",
        type=int,
        default=10,
        help="Minimum required distance between objects to be considered for spot analysis in pixels.")
		
    parser.add_argument(
        "--spot_int_thres", "-thres",
        type=float,
        default=0.4,
        help="Fraction of the maximum intensity. Used to define the minimum intensity of peaks")
    
    parser.add_argument(
        "--max_spot_size", "-s",
        type=int,
        default=80,
        help="Fraction of the maximum intensity. Used to define the minimum intensity of peaks")

    args = parser.parse_args()

    input_file = args.input_file
    nucleous_channel_raw_choice = args.nucleous_channel
    intensity_channel_choice = args.intensity_channel
    enable_spot_counting = args.enable_spot_counting
    spot_channel_choice = args.spot_channel
    spot_min_dist = args.spot_min_dist
    spot_threshold = args.spot_int_thres
    spot_size=args.max_spot_size
    

    filename_base = os.path.splitext(os.path.basename(input_file))[0]

    print(f"\n> Processing {input_file}")

    # --- Handle ND2 to TIF conversion ---
    processed_tif_filename = ""
    if input_file.lower().endswith(".nd2"):
        print(".converting nd2 to tif")
        processed_tif_filename = input_file[:-4] + '.tif'
        if os.path.exists(processed_tif_filename):
            print(f".tif file ({processed_tif_filename}) already exists")
        else:
            try:
                nd2.nd2_to_tiff(input_file, processed_tif_filename, progress=False)
                print(".tif file created in the same directory")
            except Exception as e:
                print(f"Error during ND2 to TIF conversion: {e}")
                sys.exit(1)
    elif input_file.lower().endswith(".tif"):
        processed_tif_filename = input_file
        print(".input file is already a .tif")
    else:
        print("Error: Input file must be either a .nd2 or .tif file.")
        sys.exit(1)

    # --- Read TIF and split channels ---
    try:
        im_c1, im_c2, im_c3 = readTIF(processed_tif_filename)
    except Exception as e:
        print(f"Error reading TIF file '{processed_tif_filename}': {e}")
        sys.exit(1)

    channels = {'c1': im_c1, 'c2': im_c2, 'c3': im_c3}
    valid_single_channels = ['c1', 'c2', 'c3']

    # --- Determine im_nucleous based on user choice (allowing combinations) ---
    im_nucleous = None
    selected_nuc_channels = []

    channel_parts = nucleous_channel_raw_choice.split('+')

    for part in channel_parts:
        part = part.strip().lower()
        if part in valid_single_channels:
            if channels[part] is not None:
                selected_nuc_channels.append(channels[part])
            else:
                print(f"Error: Channel '{part}' specified for nucleus segmentation is None or not available in the image data.")
                sys.exit(1)
        else:
            print(f"Error: Invalid channel or combination part '{part}' provided for nucleus segmentation.")
            print(f"Valid single channels are {', '.join(valid_single_channels)}. Combinations should be like 'c1+c2'.")
            sys.exit(1)

    if not selected_nuc_channels:
        print("Error: No valid channels were selected for nucleus segmentation.")
        sys.exit(1)

    im_nucleous = selected_nuc_channels[0]
    for i in range(1, len(selected_nuc_channels)):
        im_nucleous = im_nucleous + selected_nuc_channels[i]


    # --- Determine the image for intensity profile calculation based on user choice ---
    intensity_profile_im = None
    if channels[intensity_channel_choice] is not None:
        intensity_profile_im = channels[intensity_channel_choice]
    else:
        print(f"Error: The selected intensity channel '{intensity_channel_choice}' is not available or is None.")
        sys.exit(1)

    # --- Proceed with analysis only if both necessary images are valid ---
    
    if im_nucleous is not None and intensity_profile_im is not None:

        # Image segmentation - make binary
        print(".thresholding image intensity")
        try:
            im_binary = make_binary(
		                im_nucleous,
		                sigma=11,
		                closing_radius=3,
		                min_object_size=1000,
		                max_hole_size=1000)

        except Exception as e:
            print(f"Error during binarization (make_binary): {e}")
            sys.exit(1)

        # Apply watershed segmentation
        print(".applying watershed segmentation")
        if im_binary is not None:

            try:
                im_markers = watershed_split(im_binary,
			                    			 min_distance=40,
			                    			 clear_border=True,
			                    			 min_segment_size=1000)

            except Exception as e:
                print(f"Error during watershed segmentation (watershed_split): {e}")
                sys.exit(1)

            if im_markers is not None:
                # Get nuclear properties
                print(".getting properties of the nucleous")
                try:
                    nucleous_props = get_region_props(im_markers, im_nucleous,)
                except Exception as e:
                    print(f"Error getting region properties (get_region_props): {e}")
                    sys.exit(1)
                print("Watershed regions:", np.max(im_markers))
                print("Props detected:", len(nucleous_props))
                print("Unique labels before get_region_props:", np.unique(im_markers).max())

                
                # Compute EFC ratio
                print(".computing efc ratio")
                try:
                    efc_ratios = add_efc_ratio_to_dataframe(nucleous_props)
                except Exception as e:
                    print(f"Error computing EFC ratio: {e}")
                    sys.exit(1)

                # Get radial profiles using radial rings
                n_rings = 5
                print(f".computing radial profile with {n_rings} rings")
                try:
                    profiles = calculate_radial_profile(intensity_profile_im, im_markers, n_rings)
                except Exception as e:
                    print(f"Error calculating radial profile: {e}")
                    sys.exit(1)

                # Merge tables
                nucleous_props = nucleous_props.merge(profiles)

                # --- SPOT COUNT (ONLY IF FLAG IS ACTIVATED) --
                
                if enable_spot_counting:
                    print(".spot counting is enabled.")
                    spot_channel_im = None

                    if channels[spot_channel_choice] is not None:
                        spot_channel_im = channels[spot_channel_choice]
                    else:
                        print(f"Error: The selected spot channel '{spot_channel_choice}' is not available or is None, despite --enable_spot_counting being used.")
                        sys.exit(1)

                    if im_nucleous is not None and spot_channel_im is not None:

                        # Image segmentation - make binary
                        print(" .counting brighter spots per cell")

                        try:
                            # find spots (based on peak max function)
                            spot_coords = find_local_maxima(spot_channel_im,
															threshold=spot_threshold, 
					                                        min_dist=spot_min_dist, 
					                                        max_peaks=300,
                                                            min_area=spot_min_dist,
                                                            max_area=spot_size,
                                                            max_eccentricity=0.8)
                    
                        except Exception as e:
                            print(f"Error when using find_local_maxima: {e}")
                            sys.exit(1)

						# count spots inside each segmented cell
                        spot_counts_df, spots_coords_inside = count_spots_inside_cells(spot_coords, 
                                                                                       nucleous_props)

                        # Merge with main table
                        nucleous_props = nucleous_props.merge(spot_counts_df, how='left')

                        # Create a figure with subplots for spots
                        output_filename_img_spots = filename_base + "_spots.png"

                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                        # Plot Spot Channel
                        axes[0].imshow(spot_channel_im, cmap = 'gray');
                        axes[0].set_title('Spot Channel', fontsize = 6);
                        add_labels_to_fig(axes[0], nucleous_props, s = 4, color = 'red')
                        axes[0].axis('off')

                        # Plot Spot Segmentation
                        axes[1].imshow(spot_channel_im, cmap = 'gray');
                        axes[1].set_title('Spot Detection', fontsize = 6);
                        axes[1].plot(spots_coords_inside[:,1], 
                                     spots_coords_inside[:,0], 
                                     'o', c = 'red', markersize=3, alpha = 0.2)
                        axes[1].axis('off')
						
                        # add nuclei contours
                        contours = draw_marker_contours(nucleous_props.boundaries,
                                                        axes[1], color = 'steelblue', lw = 0.3)
		      
                        plt.tight_layout()

                        try:
                            saveFig(output_filename_img_spots)
                            print(f" .saved spot image: {output_filename_img_spots}")
                        except Exception as e:
                            print(f" error saving segmented spot image: {e}")

                        plt.close(fig)
                    else:
                        print("Warning: Spot counting enabled, but required spot image data is missing. Skipping spot analysis.")
                else:
                    print(".spot counting is disabled (use --enable_spot_counting to activate).")
                    # Se a contagem de spots não estiver ativada, garante que a coluna 'spot_count' exista com valores NaN ou 0
                    
                    #if 'spot_count' not in nucleous_props.columns:
                    #     nucleous_props['spot_count'] = np.nan # Ou 0,
                    
                # Create a figure with subplots for nucleus segmentation
                print(".creating segmented images for nucleus")
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                output_filename_img = filename_base + "_segmented.png"

                # Plot raw image
                for i, b in enumerate(nucleous_props.boundaries):
                    if b is None or len(b) == 0:
                        print(f"Warning: nucleus {i} has no boundary.")

                axes[0].imshow(im_nucleous)
                axes[0].set_title(f'Raw Image ({nucleous_channel_raw_choice})')
                axes[0].axis('off')

                # Draw contours for all watershed regions directly from the marker image
                im_markers_plot = im_markers.astype(np.int32)  # ensure safe integer type
                axes[0].contour(
                    im_markers_plot,
                    levels=np.arange(1, np.max(im_markers_plot)+1),
                    colors='crimson',
                    linewidths=0.5
)


                # Plot binary image
                axes[1].imshow(im_binary, cmap='gray')
                axes[1].set_title('Binary Image')
                axes[1].axis('off')

                # Plot watershed output
                axes[2].imshow(im_markers, cmap='nipy_spectral')
                axes[2].set_title('Watershed Segmentation')
                axes[2].axis('off')

                contours = draw_marker_contours(nucleous_props.boundaries,
                                axes[0], color='crimson', lw=1)
                print(f"Contours drawn: {len(nucleous_props.boundaries)}")


                try:
                    add_labels_to_fig(axes[2], nucleous_props, s = 4, color = 'k')
                except Exception as e:
                    print(f"Warning: Error adding labels to figure: {e}")
                    pass

                plt.tight_layout()
                try:
                    saveFig(output_filename_img)
                    print(f"Saved segmented image: {output_filename_img}")
                except Exception as e:
                    print(f"Error saving segmented image: {e}")
                plt.close(fig)


                # SAVE OUTPUT TABLE
                # Ajusta as colunas para incluir 'spot_count' se estiver presente, caso contrário, ignora
                cols = ['label','area', 'mean_intensity', 'efc_ratio','shell_ratio'] + profiles.columns.tolist()[1:-1]
                
                if 'spot_count' in nucleous_props.columns:
                    cols.append('spot_count')
                
                # Garante que a ordem das colunas seja mantida conforme especificado
                # e que apenas colunas existentes sejam selecionadas
                nucleous_props = nucleous_props[[col for col in cols if col in nucleous_props.columns]]


                output_filename_table = filename_base + "_props.csv"
                os.makedirs('output', exist_ok=True)
                try:
                    nucleous_props.to_csv(os.path.join('output', output_filename_table), index = False)
                    print(f"Saved output table: output/{output_filename_table}")
                except Exception as e:
                    print(f"Error saving output table: {e}")
                    sys.exit(1)


            else:
                print("Error: Watershed segmentation did not produce markers (im_markers is None). Cannot proceed with property extraction or plotting.")
                sys.exit(1)
        else:
            print("Error: Binary image (im_binary) was not created successfully. Cannot proceed with watershed segmentation.")
            sys.exit(1)
    else:
        print("Error: Required images for analysis (im_nucleous or intensity_profile_im) are None. Please check channel selection and input image integrity.")
        sys.exit(1)