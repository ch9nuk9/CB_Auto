import os
import glob

DATASETS=glob.glob("*/*track*/")

output_folder_name='output/'

#Create a unique output folder for each dataset
for dataset in DATASETS:
    output_folder = os.path.join(dataset, output_folder_name)  # Append 'output' to each dataset path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the folder if it doesn't exist

# Prepend '/output' to each dataset path to ensure processed data goes into the output folder
DATASETS_OUTPUT = glob.glob("*/*track*/" + output_folder_name)    

background_img = glob.glob("*/*background*")[0]

print(background_img)

print(DATASETS)

rule targets:
    input:
        chemotaxis_overview = expand("{datasets_output}chemotaxis_overview.png", datasets_output=DATASETS_OUTPUT),
        principal_components=expand("{datasets_output}principal_components.csv",datasets_output=DATASETS_OUTPUT),
        output_skel_X=expand("{datasets_output}skeleton_skeleton_X_coords.csv",datasets_output=DATASETS_OUTPUT),
        output_skel_Y=expand("{datasets_output}skeleton_skeleton_Y_coords.csv",datasets_output=DATASETS_OUTPUT),
        output_spline_X=expand("{datasets_output}skeleton_spline_X_coords.csv",datasets_output=DATASETS_OUTPUT),
        output_spline_Y=expand("{datasets_output}skeleton_spline_Y_coords.csv",datasets_output=DATASETS_OUTPUT),
        corrected_head=expand("{datasets_output}skeleton_corrected_head_coords.csv",datasets_output=DATASETS_OUTPUT),
        corrected_tail=expand("{datasets_output}skeleton_corrected_tail_coords.csv",datasets_output=DATASETS_OUTPUT),
       


rule make_contour_based_binary:
    input:
        input_img = "{datasets_output}../track.tif"
    params:
        function = "make_contour_based_binary",
        threshold = config["threshold"],
        max_value = config["max_value"],
        blur_gaussian = config["blur_gaussian"],
        blur = config["blur"],
        contour_size = config["contour_size"],
        tolerance = config["tolerance"],
        inner_contour_to_fill = config["inner_contour_to_fill"],
        substract_background = 0
    output:
        #I think I could write {params.lower_threshold} in the binary_img string, so that it is not hard coded.
        binary_img = "{datasets_output}track_mask.btf"
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            str(params.function),
            '-i', str(input.input_img),
            '-o', str(output.binary_img),
            '-blur', str(params.blur),
            '-th', str(params.threshold),
            '-max_val', str(params.max_value),
            '-cs', str(params.contour_size),
            '-t', str(params.tolerance),
            '-ics', str(params.inner_contour_to_fill),
            '-blur_gaussian',str(params.blur_gaussian),
            '-sb',str(params.substract_background),
        ])

# This rule erodes the binary mask from the background-subtracted image for a specified number of iterations and saves the eroded image.
rule erode_binary_mask:
    input:
        binary_img = "{datasets_output}track_mask.btf"
    params:
        iter =  config['iterations']
    output:
        erode_img = "{datasets_output}track_mask_eroded.btf"
    run:
        import pandas as pd
        import numpy as np
        import tifffile as tiff
        from scipy import ndimage

        with tiff.TiffFile(input.binary_img) as tif, tiff.TiffWriter(output.erode_img, bigtiff=True) as output_tif:
             for i, page in enumerate(tif.pages):
                 img = page.asarray()
                 eroded_img = ndimage.binary_erosion(img,iterations=params.iter)
                 eroded_img = eroded_img * 255
                 eroded_img = eroded_img.astype(np.uint8)
                 #eroded_img = np.array(eroded_img, dtype=np.uint8)
                 # save as a btf
                 output_tif.write(eroded_img,photometric='minisblack',contiguous=True)

rule tiff2avi:
    input:
        input_img = "{datasets_output}../track.tif"
    params:
        function = "tiff2avi",
        fourcc = config["fourcc"],
        fps = config["fps"]
    output:
        avi = "{datasets_output}track.avi"
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            params.function,
            '-i', str(input.input_img),
            '-o',str(output.avi),
            '-fourcc', str(params.fourcc),
            '-fps', str(params.fps),
        ])

rule dlc_analyze_videos:
    input:
        avi = "{datasets_output}track.avi"
    params:
        dlc_model_configfile_path = config["dlc_model_configfile_path"],
        network_string = config["network_string"],
    #TODO: run conda properly, instead of activating it in the shell. # conda: "/scratch/neurobiology/zimmer/.conda/envs/HR_tracker"
    output:
        hdf5_file = "{datasets_output}track"+config["network_string"]+".h5",
        csv= "{datasets_output}track"+config["network_string"]+".csv"
    shell:
        """
        source /lisc/app/conda/miniconda3/bin/activate /lisc/scratch/neurobiology/zimmer/.conda/envs/deeplabcut_imutils 
        python -c "from imutils.src import DLC_analyze_videos; DLC_analyze_videos.main(['--path_config_file', '{params.dlc_model_configfile_path}', '--videofile_path', '{input.avi}'])"
        """

rule create_centerline:
    input:
        #TODO: Try ancient() to see if I can avoid re-running when the script is modified
        input_binary_img = "{datasets_output}track_mask_eroded.btf",
        hdf5_file = "{datasets_output}track"+config["network_string"]+".h5"
        #hdf5_file = "{datasets}raw_stack"+config["network_string"]+".h5" #"AVG_background_subtracted_normalisedDLC_resnet50_wbfm_nose_tailJan4shuffle1_1030000.h5"

    params:
        csv_output_path = "{datasets_output}",  # This cant go into inputs because it is not a file
        number_of_neighbours = "1",
        nose = config['nose'],
        tail = config['tail'],
        num_splines = config['num_splines'],
        fill_with_DLC = "1",  # TODO: Improve so argparse takes boolean
        min_worm_lenght = "40" #set min lenght what is recognized as worm in create centerline

    output :
        output_skel_X = "{datasets_output}skeleton_skeleton_X_coords.csv",
        output_skel_Y = "{datasets_output}skeleton_skeleton_Y_coords.csv",
        output_spline_K = "{datasets_output}skeleton_spline_K.csv",
        output_spline_X = "{datasets_output}skeleton_spline_X_coords.csv",
        output_spline_Y = "{datasets_output}skeleton_spline_Y_coords.csv",
        corrected_head = "{datasets_output}skeleton_corrected_head_coords.csv",
        corrected_tail = "{datasets_output}skeleton_corrected_tail_coords.csv"
    run:
        from centerline_behavior_annotation.centerline.dev import head_and_tail

        head_and_tail.main([
            '-i', str(input.input_binary_img),
            '-h5', str(input.hdf5_file),
            '-o', str(params.csv_output_path),
            '-nose', str(params.nose),
            '-tail', str(params.tail),
            '-num_splines', str(params.num_splines),
            '-n', str(params.number_of_neighbours),
            '-dlc', str(params.fill_with_DLC),
            '-mw', str(params.min_worm_lenght),
        ])

rule invert_curvature_sign:
    input:
        spline_K = "{datasets_output}skeleton_spline_K.csv",
    params:
        stage_pos= "{datasets_output}../worm_config.yaml",
    output:
        spline_K_signed = "{datasets_output}skeleton_spline_K_signed.csv"
    run:        
        import yaml
        from centerline_behavior_annotation.curvature.src import invert_curvature_sign
        import os
        import sys

        # Construct the path to the config file one level up from params.path using ../
        config_path = params.stage_pos

        print(config_path)

        try:
            # Read the config file
            with open(config_path, 'r') as stream:
                worm_config = yaml.safe_load(stream)
            
            # Extract the 'ventral' parameter from the config
            ventral = worm_config.get('ventral', 'left')
            print(ventral)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error: {e}. Setting 'ventral' to 'left' by default.")
            ventral = 'left'

        # Call the invert_curvature_sign function with the correct parameters
        invert_curvature_sign.main_benjamin([
            '--spline_K_path', str(input.spline_K),
            '--ventral', str(ventral),
            '--output_file_path', str(output.spline_K_signed),
        ])

rule annotate_behaviour:
    input:
        curvature_file = "{datasets_output}skeleton_spline_K_signed.csv"
    params:
        pca_model_path = config["pca_model_path"],
        initial_segment = config["initial_segment"],
        final_segment = config["final_segment"],
        window = config["window"],
        upper_threshold = config["upper_threshold"],
        lower_threshold = config["lower_threshold"]
    output:
        principal_components = "{datasets_output}principal_components.csv",
        behaviour_annotation = "{datasets_output}beh_annotation.csv"

    run:
        from centerline_behavior_annotation.curvature.src import annotate_reversals_snakemake
        
        annotate_reversals_snakemake.main([
            '-i', str(input.curvature_file),
            '-pca', str(params.pca_model_path),
            '-i_s', str(params.initial_segment),
            '-f_s', str(params.final_segment),
            '-win', str(params.window),
            '-o_bh', str(output.behaviour_annotation),
            '-o_pc', str(output.principal_components),
            '--upper_threshold', str(params.upper_threshold),
            '--lower_threshold', str(params.upper_threshold),
        ])

rule annotate_turns:
    input:
        curvature_file="{datasets_output}skeleton_spline_K_signed.csv"
    params:
        threshold = config["turn_threshold"],
        initial_segment=config["initial_segment"],
        final_segment=config["final_segment"],
        window=config["window"],
    output:
        turn_annotation="{datasets_output}turn_annotation.csv"

    run:
        from centerline_behavior_annotation.curvature.src import annotate_turns_snakemake

        annotate_turns_snakemake.main([
            '-input', str(input.curvature_file),
            '-t', str(params.threshold),
            '-i_s', str(params.initial_segment),
            '-f_s', str(params.final_segment),
            '-avg_window', str(params.window),
            '-bh', str(output.turn_annotation),
        ])


rule create_plots2:
    input:
        behaviour_annotation="{datasets_output}beh_annotation.csv",
        turn_annotation="{datasets_output}turn_annotation.csv",
        spline_K= "{datasets_output}skeleton_spline_K_signed.csv",
        skeleton_spline_X_coords= "{datasets_output}skeleton_spline_X_coords.csv",
        skeleton_spline_Y_coords= "{datasets_output}skeleton_spline_Y_coords.csv",
        conc_gradient_array= "/lisc/scratch/neurobiology/zimmer/schaar/diffusion/benzaldehyde/conc_array_benzaldehyde.npy",
        distance_array= "/lisc/scratch/neurobiology/zimmer/schaar/diffusion/benzaldehyde/distance_array_benzaldehyde.npy",
    params:
        factor_px_to_mm = config["factor_px_to_mm"],
        fps = config["fps"],
        worm_pos= "{datasets_output}../track.txt",
        top_left_pos = config["top_left"],
        odor_pos  = config["odor_pos"],
        tif_path = "{datasets_output}../track.tif",
    output:
        #angle_animation= "{datasets_output}angle_animation.avi",
        chemotaxis_overview= "{datasets_output}chemotaxis_overview.png",
        #worm_movie = "{datasets_output}worm_movie.avi",
    run:
        from chemotaxis_analysis_high_res import initialize_load_files_population
        import tifffile as tiff
        from imutils.scopereader import MicroscopeDataReader
        import dask.array as da
        import numpy as np

        def get_video_resolution(tif_path):
            
            # List all files in the directory
            reader_obj = MicroscopeDataReader(tif_path, as_raw_tiff=True, raw_tiff_num_slices=1)
            tif = da.squeeze(reader_obj.dask_array)
            
            for i, img in enumerate(tif):
                # Read the first frame
                
                first_frame = np.array(img)
                
                # Get the x and y dimensions
                video_resolution_y, video_resolution_x = first_frame.shape

                break

            
            return video_resolution_x, video_resolution_y

        tif_path = str(params.tif_path)
        video_resolution_x, video_resolution_y = get_video_resolution(tif_path)

        print(video_resolution_x, video_resolution_y)
        

        initialize_load_files_population.main([
            '--beh_annotation', str(input.behaviour_annotation),
            '--skeleton_spline', str(input.spline_K),
            '--worm_pos', str(params.worm_pos),
            '--skeleton_spline_X_coords', str(input.skeleton_spline_X_coords),
            '--skeleton_spline_Y_coords', str(input.skeleton_spline_Y_coords),
            '--factor_px_to_mm', str(params.factor_px_to_mm),
            '--video_resolution_x', str(video_resolution_x),
            '--video_resolution_y', str(video_resolution_y),
            '--fps', str(params.fps),
            '--conc_gradient_array', str(input.conc_gradient_array),
            '--distance_array', str(input.distance_array),
            '--turn_annotation', str(input.turn_annotation),
            '--top_left_pos', str(params.top_left_pos),
            '--odor_pos', str(params.odor_pos),
        ])
