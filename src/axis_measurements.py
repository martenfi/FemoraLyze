# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: axis_measurements.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script computes NSA and FO based on the center coordinates of 
#              the femur.
# ==================================================================================



import numpy as np
import pandas as pd
import os
from src.utils.utils import print_module_start, check_save_output



class AxisMeasurements:
    def __init__(self, config):
        self.config = config
        self.voxel_spacing = config.voxel_spacing_mm  # mm

    def neck_shaft_angle(self, centers: np.array) -> float:
        """
        Calculate the neck-shaft angle (NSA) based on the center coordinates.

        Parameters:
            centers (np.array): Numpy array containing the center coordinates for each subpointcloud of the pointcloud.
        
        Returns:
            nsa (float): Neck-Shaft Angle in degrees.
        """
        center_head = centers[0]
        center_neck = centers[1]
        center_trochanter = centers[2]
        center_shaft = centers[3]

        # Calculate the vectors
        vector_head_neck = center_neck - center_head
        vector_shaft_trochanter = center_trochanter - center_shaft

        # Calculate neck-shaft angle
        nsa = np.arccos(np.dot(vector_head_neck, vector_shaft_trochanter) / (np.linalg.norm(vector_head_neck) * np.linalg.norm(vector_shaft_trochanter)))

        # Convert to degrees
        nsa = np.degrees(nsa)  
        return nsa


    def femural_offset(self, centers: np.array) -> float:
        """
        Calculate the femoral offset (FO) based on the center coordinates.

        Parameters:
            centers (np.array): Numpy array containing the center coordinates for each subpointcloud of the pointcloud.

        Returns:
            fo (float): Femoral Offset in mm.
        """
        center_head = centers[0]
        center_trochanter = centers[2]
        center_shaft = centers[3]

        # Calculate the vector
        vector_shaft_trochanter = center_trochanter - center_shaft

        # Calculate femoral offset
        fo = np.linalg.norm(np.cross(vector_shaft_trochanter, center_head - center_shaft)) / np.linalg.norm(vector_shaft_trochanter)

        # Convert to mm
        voxel_spacing = self.voxel_spacing
        fo = fo * voxel_spacing  #mm
        return fo


    def process(self, centers: np.array) -> None:
        """
        Calculate the neck-shaft angle and femoral offset based on the center coordinates.

        Parameters:
            centers (np.array): Numpy array containing the center coordinates for each subpointcloud of the pointcloud.

        Returns:
            nsa (float): Neck-Shaft Angle in degrees.
            f0 (float): Femoral Offset in mm.
        """
        print_module_start("Axis Measurements")
        nsa = self.neck_shaft_angle(centers)
        fo = self.femural_offset(centers)
        # Create a pandas DataFrame to store the results
        df = pd.DataFrame({
            "Measurement": ["Neck-Shaft Angle (NSA)", "Femoral Offset (FO)"],
            "Value": [nsa, fo],
            "Unit": ["degrees", "mm"]
        })

        print(df.to_string(index=False))

        if check_save_output(self.config, "metrics", self.config.save_results.axis_measurements):
            axis_measurements_dir = os.path.join(self.config.experiment_dir, "metrics", "axis_measurements.csv")
            df.to_csv(axis_measurements_dir, index=False)


        return 



if __name__ == "__main__":
    pass