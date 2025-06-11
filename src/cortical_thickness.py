# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: cortical_thickness.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script computes the cortical thickness based on the Hildebrand 
#              algorithm.
# ==================================================================================



import numpy as np
import pandas as pd
import time
import os
from scipy.ndimage import distance_transform_edt
from src.bone_metrics import BoneMetrics
from src.utils.utils import print_module_start, check_save_output


class CorticalThickness:
    def __init__(self, config):
        self.config = config
        self.voxel_size = config.voxel_spacing_mm

    def process(self, binary_mask: np.ndarray) -> float:
        """ 
        Computes the cortical thickness of the femur neck using the Hildebrand algorithm.

        Parameters:
            binary_mask (np.ndarray): A binary mask of the femur neck region.
            voxel_size (float): The size of a voxel in mm.

        Returns:
            float: The cortical thickness in mm.
        """
        print_module_start("Cortical Thickness")
        bm = BoneMetrics(self.config)
        # according to hildebrand algorithm
        if np.sum(binary_mask) == 0:
            return 0, 0, 0
        # Trabecular thickness
        start_time = time.time()
        distance_map = distance_transform_edt(binary_mask)
        print(f"Step 1 - Distance map computation: {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        distance_ridge = bm.compute_distance_ridge(distance_map)
        print(f"Step 2 - Distance ridge computation: {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        local_thickness = bm.compute_local_thickness_from_distance_ridge(distance_ridge)
        print(f"Step 3 - Local thickness computation: {time.time() - start_time:.4f} seconds")


        start_time = time.time()
        local_thickness_updated = bm.replace_surface_voxels(local_thickness, distance_map)
        print(f"Step 4 - Replace surface voxels: {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        cortical_thickness_neck = np.mean(local_thickness_updated[local_thickness_updated > 0])
        cortical_thickness_neck_mm = cortical_thickness_neck * self.voxel_size
        print(f"Step 5 - Cortical thickness computation: {time.time() - start_time:.4f} seconds")

        df = pd.DataFrame([["Femur Neck", cortical_thickness_neck_mm]], columns=["Region", "Cortical Thickness (mm)"])
        print(df.to_string(index=False))
        
        if check_save_output(self.config, "metrics", self.config.save_results.cortical_thickness):
            cortical_thickness_dir = os.path.join(self.config.experiment_dir, "metrics", "cortical_thickness.csv")
            df.to_csv(cortical_thickness_dir, index=False)

        return cortical_thickness_neck_mm




if __name__ == "__main__":
    pass


