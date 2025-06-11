# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: region_extraction.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script extracts the regions of the femur CT based on the 
#              bone region mask.
# ==================================================================================



from src.utils.utils import load_nifti, save_nifti, print_module_start, check_save_output
import numpy as np
import os




class RegionExtraction:
    def __init__(self, config):
        self.config = config
        self.padding = config.region_extraction.padding
        self.label_meanings = {int(k): v for k, v in self.config.label_meanings._to_dict().items() if k.isdigit()}

    def __extract_bone_region(self, binary_mask: np.ndarray, voxel_array_region: np.ndarray) -> np.ndarray:
        """
        Extracts a region from the voxel array based on the binary mask and applies padding.

        Parameters:
            binary_mask (np.ndarray): A binary mask indicating the region to extract.
            voxel_array_region (np.ndarray): The voxel array from which to extract the region.

        Returns:
            np.ndarray: The extracted region with padding applied.
        """
        padding = self.padding
        region_indices = np.where(binary_mask > 0)
        min_x, max_x = np.min(region_indices[0]), np.max(region_indices[0])
        min_y, max_y = np.min(region_indices[1]), np.max(region_indices[1])
        min_z, max_z = np.min(region_indices[2]), np.max(region_indices[2])

        min_x = max(0, min_x - padding)
        max_x = min(voxel_array_region.shape[0], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(voxel_array_region.shape[1], max_y + padding)
        min_z = max(0, min_z - padding)
        max_z = min(voxel_array_region.shape[2], max_z + padding)

        cut_out_region = voxel_array_region[min_x:max_x, min_y:max_y, min_z:max_z]

        return cut_out_region


    def process(self, bone_region_mask: np.ndarray, voxel_array_bone: np.ndarray, meta_dict: dict) -> list:
        """
        Processes the bone region mask and extracts individual bone regions.

        Parameters:
            bone_region_mask (np.ndarray): The mask indicating different bone regions.
            voxel_array_bone (np.ndarray): The voxel array of the bone.
            meta_dict (dict): Metadata dictionary for saving NIfTI files.

        Returns:
            list: A list of extracted bone regions.
        """
        print_module_start("Region Extraction")
        label_meanings_2 = {key: value.replace(" ", "_") for key, value in self.label_meanings.items()}

        bone_regions = []
        for label in self.label_meanings.keys():
            if label == 0:
                continue
            else:
                binary_mask = bone_region_mask == label
                voxel_array_region = np.where(binary_mask, voxel_array_bone, 0)
                cut_out_region = self.__extract_bone_region(binary_mask, voxel_array_region)
                bone_regions.append(cut_out_region)
                if check_save_output(self.config, "regions", self.config.save_results.regions):
                    regions_dir = os.path.join(self.config.experiment_dir, "regions")
                    file_name = f"Region_{label_meanings_2[label]}.nii.gz"
                    save_nifti(cut_out_region, meta_dict, regions_dir, file_name)
        return bone_regions




if __name__ == "__main__":
    pass

    

    