# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: roi_extraction.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script extracts the regions of interest (ROIs) from the femur CT 
#              based on the center coordinates.
# ==================================================================================



import numpy as np
import os
from src.utils.utils import save_nifti, print_module_start, check_save_output


class RoiExtraction:
    def __init__(self, config):
        self.config = config
        self.roi_size = config.roi_extraction.roi_size

    def __compute_roi_boundaries(self, coordinates: np.array, roi_size: int) -> list:
        """
        Calculates the boundaries of a region of interest (ROI) around a given center coordinate.

        Parameters:
            coordinates (np.array): Center coordinates of the ROI.
            roi_size (int): Size of the ROI.

        Returns:
            boundaries (list): List of the boundaries of the ROI [x0, x1, y0, y1, z0, z1].
        """
        coordinates = np.round(coordinates, 0).astype(int)
        half_roi = roi_size / 2
        x0 = coordinates[0] - half_roi
        x1 = coordinates[0] + half_roi
        y0 = coordinates[1] - half_roi
        y1 = coordinates[1] + half_roi
        z0 = coordinates[2] - half_roi
        z1 = coordinates[2] + half_roi
        boundaries = [int(x0), int(x1), int(y0), int(y1), int(z0), int(z1)]
        return boundaries


    def __cut_roi(self, voxel_array: np.array, boundaries: list) -> np.array:
        """
        Cuts a region of interest (ROI) from a voxel array based on the given boundaries.

        Parameters:
            voxel_array (np.array): Voxel array as the source.
            boundaries (list): List of the boundaries of the ROI [x0, x1, y0, y1, z0, z1].

        Returns:
            voxel_array_roi (np.array): Voxel array of the ROI extracted from the source voxel_array.
        """
        x0, x1, y0, y1, z0, z1 = boundaries
        voxel_array_roi = voxel_array[x0:x1, y0:y1, z0:z1]
        return voxel_array_roi


    def process(self, voxel_array_bone_structure: np.array, coordinates: np.array, meta_dict: dict) -> list:
        """
        Extracts all region of interests (ROIs) from sr voxel array based on the given boundaries.

        Parameters:
            voxel_array_bone_structure (np.array): Voxel array of the bone structure.
            coordinates (np.array): Center coordinates of the ROIs.
            meta_dict (dict): Metadata dictionary containing information about the voxel array.
        
        Returns:
            all_rois (list): List of extracted ROIs as voxel arrays.
        """
        print_module_start("ROI Extraction")
        roi_size = self.roi_size
        print(f"ROI size: {roi_size}")

        region_coordinates = [coordinates[0], coordinates[1], coordinates[2]] # Only ROIs for head, neck and trochanter are extracted, shaft is not relevant here
        boundaries = [self.__compute_roi_boundaries(coordinate, roi_size) for coordinate in region_coordinates]

        all_rois = []
        roi_names = ["ROI_Head", "ROI_Neck", "ROI_Trochanter"] # Only ROIs for head, neck and trochanter are extracted, shaft is not relevant here
        for i in range(len(roi_names)):
            roi = self.__cut_roi(voxel_array_bone_structure, boundaries[i])
            all_rois.append(roi)

            if check_save_output(self.config, "rois", self.config.save_results.rois):
                roi_dir = os.path.join(self.config.experiment_dir, "rois")
                file_name_nii = f"{roi_names[i]}_{roi_size}.nii.gz"
                save_nifti(roi, meta_dict, roi_dir, file_name_nii)
        
        return all_rois

    

if __name__ == "__main__":
    pass

