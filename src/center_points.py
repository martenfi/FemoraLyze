# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: center_points.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script computes the center coordinates of the femur based on the
#              regions of the femur.
# ==================================================================================



import open3d as o3d
import numpy as np
import pandas as pd
import os
from src.utils.utils import print_module_start, check_save_output



class CenterPoints:
    def __init__(self, config):
        self.config = config
        self.label_colors = {int(k): v for k, v in self.config.label_colors._to_dict().items() if k.isdigit()}
        self.label_meanings = {int(k): v for k, v in self.config.label_meanings._to_dict().items() if k.isdigit()}

    def __select_subpointcloud(self, pcl: o3d.geometry.PointCloud, label: int, label_colors: dict) -> o3d.geometry.PointCloud: #redundant as it is already in BoneRegionMask
        """
        Select subpointcloud from a pointcloud object based on the label encoded in the color attribute.

        Parameters:
            pcl (o3d.geometry.PointCloud): Pointcloud object.
            label (int): Label of the subpointcloud.
            label_colors (dict): Dictionary containing the label colors.
        
        Returns:
            sub_pcl (o3d.geometry.PointCloud): Subpointcloud object for the given label.
        """
        label_color = label_colors[label]
        points = np.asarray(pcl.points)[np.all(np.asarray(pcl.colors) == label_color, axis=1)]
        colors = np.asarray(pcl.colors)[np.all(np.asarray(pcl.colors) == label_color, axis=1)]
        normals = np.asarray(pcl.normals)[np.all(np.asarray(pcl.colors) == label_color, axis=1)]

        sub_pcl = o3d.geometry.PointCloud()
        sub_pcl.points = o3d.utility.Vector3dVector(points)
        sub_pcl.colors = o3d.utility.Vector3dVector(colors)
        sub_pcl.normals = o3d.utility.Vector3dVector(normals)
        return sub_pcl


    def process(self, pcl: o3d.geometry.PointCloud) -> np.array:
        """
        Extracts the center coordinates of the subpointclouds for the given pointcloud.

        Parameters:
            pcl (o3d.geometry.PointCloud): Pointcloud object with labels encoded in the color attribute.
        
        Returns:
            centers (np.array): Numpy array containing the center coordinates for each subpointcloud of the pointcloud.
        """
        print_module_start("Center Coordinates")
        centers = []

        pcl_head = self.__select_subpointcloud(pcl, 2, self.label_colors)
        pcl_neck = self.__select_subpointcloud(pcl, 4, self.label_colors)
        pcl_trochanter = self.__select_subpointcloud(pcl, 1, self.label_colors)
        pcl_shaft = self.__select_subpointcloud(pcl, 3, self.label_colors)

        center_head = pcl_head.get_center()
        center_neck = pcl_neck.get_center()
        center_trochanter = pcl_trochanter.get_center()
        center_shaft = pcl_shaft.get_center()
        centers = [center_head, center_neck, center_trochanter, center_shaft]
        
        # Create a pandas DataFrame for the center coordinates
        data = []
        for label, meaning in self.label_meanings.items():
            if label == 0:  # Background
                data.append([meaning, label, 0, 0, 0])
            else:
                center = centers[label - 1]
                data.append([meaning, label, center[0], center[1], center[2]])

        df = pd.DataFrame(data, columns=["Label Meaning", "Label Number", "X", "Y", "Z"])
        print(df.to_string(index=False))
        
        if check_save_output(self.config, "metrics", self.config.save_results.center_points):
            center_points_dir = os.path.join(self.config.experiment_dir, "metrics", "center_coordinates.csv")
            df.to_csv(center_points_dir, index=False)

        return np.array(centers)






if __name__ == "__main__":
    pass
