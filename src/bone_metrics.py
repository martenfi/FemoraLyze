# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: bone_metrics.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script computes the bone metrics (BV/TV, Tb.Th, Tb.Sp, Tb.N)
#              of the femur ROIs using the Hildebrand algorithm.
# ==================================================================================


import numpy as np
import pandas as pd
import os
from scipy.ndimage import distance_transform_edt
from numba import cuda as nbcuda
from src.utils.utils import print_module_start, check_save_output



class BoneMetrics:
    def __init__(self, config):
        self.config = config
        self.voxel_spacing_mm = config.voxel_spacing_mm

    def compute_distance_ridge(self, distance_map: np.ndarray) -> np.ndarray:
        """
        Compute a distance ridge map from a distance map using a cubic neighborhood.

        Parameters:
            distance_map (np.ndarray): 3D array containing distance values derived from the binary mask via euclidean distance transform.
        
        Returns:
            distance_ridge (np.ndarray): 3D array containing distance ridge values.
        """
        # Get input shape
        d, h, w  = distance_map.shape
        distance_ridge = np.zeros_like(distance_map)
        # Determine the maximum distance value in the distance map
        dist_max = np.max(distance_map)
        r_sq_max = int(dist_max ** 2 + 0.5) + 1
        # Create occurrence array to track which squared distances are present
        occurs = np.zeros(r_sq_max, dtype=bool)
        # Mark presence of squared distance values in occurs array
        for z in range(d):
            for y in range(h):
                for x in range(w):
                    skind = distance_map[z, y, x]
                    occurs[int(skind ** 2 + 0.5)] = True
        # Count the number of unique squared distance values
        num_radii = np.sum(occurs)
        # Create index mappings for squared distances
        dist_sq_index = np.zeros(r_sq_max, dtype=int)
        dist_sq_values = np.zeros(num_radii, dtype=int)
        # Populate the squared distance index mapping
        ind_ds = 0
        for i in range(r_sq_max):
            if occurs[i]:
                dist_sq_index[i] = ind_ds
                dist_sq_values[ind_ds] = i
                ind_ds += 1

        # Function to create a distance template for ridge detection
        def create_template(dist_sq_values: np.ndarray) -> list:
            """
            Create a template for squared distances to be used in ridge detection.

            Parameters:
                dist_sq_values (np.ndarray): Array of squared distance values.

            Returns:
                list: A list of arrays representing the squared distances for each cubic neighborhood.
            """
            t = [scan_cube(1, 0, 0, dist_sq_values),
                scan_cube(1, 1, 0, dist_sq_values),
                scan_cube(1, 1, 1, dist_sq_values)]
            return t

        # Function to scan cubic neighborhood for ridge points
        def scan_cube(dx: int, dy: int, dz: int, dist_sq_values: np.ndarray) -> np.ndarray:
            """
            Scan a cubic neighborhood for ridge points based on squared distances.

            Parameters:
                dx (int): Offset in the x-direction.
                dy (int): Offset in the y-direction.
                dz (int): Offset in the z-direction.
                dist_sq_values (np.ndarray): Array of squared distance values.
            
            Returns:
                np.ndarray: An array of squared distances for the cubic neighborhood.
            """
            num_radii = len(dist_sq_values)
            r1_sq = np.zeros(num_radii, dtype=int)
            if (dx == 0) and (dy == 0) and (dz == 0):
                r1_sq.fill(np.iinfo(np.int32).max)  # Set to max int value
            else:
                dx_abs, dy_abs, dz_abs = -abs(dx), -abs(dy), -abs(dz)
                for r_sq_ind in range(num_radii):
                    r_sq = dist_sq_values[r_sq_ind]
                    max_val = 0
                    r = 1 + int(np.sqrt(r_sq))  # compute radius
                    for k in range(r + 1):
                        scank = k * k
                        dk = (k - dz_abs) ** 2
                        for j in range(r + 1):
                            scankj = scank + j * j
                            if scankj <= r_sq:
                                i_plus = int(np.sqrt(r_sq - scankj)) - dx_abs
                                dkji = dk + (j - dy_abs) ** 2 + i_plus ** 2
                                if dkji > max_val:
                                    max_val = dkji
                    r1_sq[r_sq_ind] = max_val
            return r1_sq

        # Generate template for squared distances
        r_sq_template = create_template(dist_sq_values)
        # Iterate through distance map to determine ridge points
        for z in range(d):
            for y in range(h):
                for x in range(w):
                    skind = distance_map[z, y, x]
                    if skind > 0:
                        not_ridge_point = False
                        sk0_sq = int(skind ** 2 + 0.5)
                        sk0_sq_ind = dist_sq_index[sk0_sq]
                        # Check the 3x3x3 neighborhood for ridge conditions
                        for dz in range(-1, 2):
                            k1 = z + dz
                            if 0 <= k1 < d:
                                for dy in range(-1, 2):
                                    j1 = y + dy
                                    if 0 <= j1 < h:
                                        for dx in range(-1, 2):
                                            i1 = x + dx
                                            if 0 <= i1 < w:
                                                if dx != 0 or dy != 0 or dz != 0:
                                                    sk1_sq = int(distance_map[k1, j1, i1] ** 2 + 0.5)
                                                    # Check if neighboring point invalidates ridge condition
                                                    if sk1_sq >= r_sq_template[abs(dz) + abs(dy) + abs(dx) - 1][sk0_sq_ind]:
                                                        not_ridge_point = True
                                            if not_ridge_point:
                                                break
                                        if not_ridge_point:
                                            break
                                if not_ridge_point:
                                    break
                        # If it is a ridge point, retain the distance value
                        if not not_ridge_point:
                            distance_ridge[z, y, x] = skind
        return distance_ridge


    def compute_local_thickness_from_distance_ridge(self, distance_ridge: np.ndarray) -> np.ndarray:
        """
        Compute local thickness from a distance ridge map using GPU acceleration with Numba.
        
        Parameters:
            distance_ridge (np.ndarray): 3D array containing distance ridge values.
        
        Returns:
            local_thickness (np.ndarray): 3D array containing local thickness values.
        """

        # Define the kernel function that will run on the GPU
        @nbcuda.jit

        def __compute_local_thickness_kernel(local_thickness, ridge_points, ridge_values, shape):
            """
            Compute the local thickness kernel function.

            Parameters:
                local_thickness (nbcuda.float32[:,:,:]): 3D array to store local thickness values.
                ridge_points (nbcuda.int32[:,:]): 2D array containing indices of ridge points.
                ridge_values (nbcuda.float32[:]): 1D array containing distance values for ridge points.
                shape (nbcuda.int32[:]): 1D array containing the shape of the input distance ridge array.

            Returns:
                None
            """
            
            # Get thread indices
            x, y, z = nbcuda.grid(3)
            # Check if the thread is within bounds of the grid
            if x >= shape[0] or y >= shape[1] or z >= shape[2]:
                return
                # Loop through all ridge points in the grid
            for idx in range(ridge_points.shape[0]):
                i, j, k = ridge_points[idx]
                h_l = ridge_values[idx]
                # Compute squared distance from the current point (x, y, z) to ridge point (i, j, k)
                dist_sq = (x - i) ** 2 + (y - j) ** 2 + (z - k) ** 2
                # If within the local thickness range, update the local thickness value
                if dist_sq < h_l ** 2:
                    local_thickness[x, y, z] = max(local_thickness[x, y, z], h_l)

        # Convert the input array to a NumPy array and prepare it for the GPU
        distance_ridge_gpu = np.asarray(distance_ridge, dtype=np.float32)
        shape = distance_ridge_gpu.shape
        # Create an empty array for the local thickness result
        local_thickness_gpu = np.zeros(shape, dtype=np.float32)
        # Get list of nonzero distance ridge points
        ridge_points = np.array(np.nonzero(distance_ridge_gpu > 0)).T  # Shape: [N, 3]
        ridge_values = distance_ridge_gpu[ridge_points[:, 0], ridge_points[:, 1], ridge_points[:, 2]]
        # Allocate memory for the data on the device
        d_local_thickness = nbcuda.to_device(local_thickness_gpu)
        d_ridge_points = nbcuda.to_device(ridge_points)
        d_ridge_values = nbcuda.to_device(ridge_values)
        d_shape = nbcuda.to_device(np.array(shape, dtype=np.int32))
        # Define block and grid dimensions for kernel execution
        threads_per_block = (8, 8, 8)  # Adjust based on your GPU and data size
        blocks_per_grid = (
            int(np.ceil(shape[0] / threads_per_block[0])),
            int(np.ceil(shape[1] / threads_per_block[1])),
            int(np.ceil(shape[2] / threads_per_block[2]))
        )
        # Launch the kernel
        __compute_local_thickness_kernel[blocks_per_grid, threads_per_block](
            d_local_thickness, d_ridge_points, d_ridge_values, d_shape
        )
        # Copy the result back to the host (CPU)
        local_thickness = d_local_thickness.copy_to_host()
        return 2 * local_thickness


    def replace_surface_voxels(self, local_thickness: np.ndarray, distance_map: np.ndarray) -> np.ndarray:
        """
        Replaces surface artifacts by replacing surface values by the average of the non-surface, structure voxels.

        Parameters:
            local_thickness (np.ndarray): 3D array containing local thickness values.
            distance_map (np.ndarray): 3D array containing distance values derived from the binary mask via euclidean distance transform.
        
        Returns:
            local_thickness_updated (np.ndarray): 3D array containing updated local thickness values.
        """
        surface = np.zeros_like(distance_map)
        surface = distance_map == 1
        structure = (distance_map > 1)
        local_thickness_updated = local_thickness.copy()
        # Iterate over surface voxels
        surface_indices = np.argwhere(surface)
        for idx in surface_indices:
            u, v, w = idx
            # Extract the 3x3x3 neighborhood
            neighborhood = local_thickness[max(0, u - 1):u + 2, max(0, v - 1):v + 2, max(0, w - 1):w + 2]
            structure_neighborhood = structure[max(0, u - 1):u + 2, max(0, v - 1):v + 2, max(0, w - 1):w + 2]
            # Mask to exclude surface voxels and background
            non_surface_mask = (neighborhood > 0) & structure_neighborhood
            # Compute the average of the non-surface, structure voxels
            if np.any(non_surface_mask):
                local_thickness_updated[u, v, w] = np.mean(neighborhood[non_surface_mask])
        return local_thickness_updated


    def __hildebrand_algorithm(self, binary_mask: np.ndarray, voxel_size: float) -> tuple:
        """
        Compute the trabecular thickness, trabecular separation and trabecular number using the Hildebrand algorithm (based on original paper, ormir-xct and Bone-J).

        Parameters:
            binary_mask (np.ndarray): 3D array containing the binary mask of the ROI.
            voxel_size (float): The size of the voxel in mm.
        
        Returns:
            Tb_Th_mm (float): The trabecular thickness in mm.
            Tb_Sp_mm (float): The trabecular separation in mm.
            Tb_N_mm (float): The trabecular number in mm^-1.

        """
        if np.sum(binary_mask) == 0:
            return 0, 0, 0
        # Trabecular thickness
        distance_map = distance_transform_edt(binary_mask)
        distance_ridge = self.compute_distance_ridge(distance_map)
        local_thickness = self.compute_local_thickness_from_distance_ridge(distance_ridge)
        local_thickness_updated = self.replace_surface_voxels(local_thickness, distance_map)
        Tb_Th = np.mean(local_thickness_updated[local_thickness_updated > 0])
        Tb_Th_mm = Tb_Th * voxel_size
        # Trabecular separation
        binary_mask_inv = np.logical_not(binary_mask)
        distance_map_inv = distance_transform_edt(binary_mask_inv)
        distance_ridge_inv = self.compute_distance_ridge(distance_map_inv)
        local_thickness_inv = self.compute_local_thickness_from_distance_ridge(distance_ridge_inv)
        local_thickness_inv_updated = self.replace_surface_voxels(local_thickness_inv, distance_map_inv)
        Tb_Sp = np.mean(local_thickness_inv_updated[local_thickness_inv_updated > 0])
        Tb_Sp_mm = Tb_Sp * voxel_size
        # Trabecular Number
        Tb_N_mm = 1 / (Tb_Th_mm + Tb_Sp_mm)
        return Tb_Th_mm, Tb_Sp_mm, Tb_N_mm


    def __bvtv(self, roi_array_binary: np.ndarray) -> float:
        """
        Computes the Bone Volume Fraction (BV/TV).

        Parameters:
            roi_array_binary (numpy.ndarray): 3D-Array of the binary ROI.
        
        Returns:
            bvtv (float): Bone Volume Fraction
        """
        bvtv = np.sum(roi_array_binary) / np.size(roi_array_binary)
        return bvtv


    def process(self, rois_binary: list) -> None:
        """
        Compute the bone metrics for the ROIs.

        Parameters:
            rois_binary (list): List of binary ROIs.

        Returns:
            roi_bone_metrics_patient (list): List containing the bone_metrics of the patient
        """
        print_module_start("Bone Metrics")
        roi_bone_metrics_patient = []

        for roi in rois_binary:
            Tb_Th_mm, Tb_Sp_mm, Tb_N_mm = self.__hildebrand_algorithm(roi, self.voxel_spacing_mm)
            bvtv_value = self.__bvtv(roi)
            roi_bone_metrics_patient.extend([bvtv_value, Tb_Th_mm, Tb_Sp_mm, Tb_N_mm])
        
        roi_names = ["ROI_Head", "ROI_Neck", "ROI_Trochanter"]
        df = pd.DataFrame({
            "ROI": roi_names,
            "BV/TV": roi_bone_metrics_patient[0::4],
            "Tb.Th (mm)": roi_bone_metrics_patient[1::4],
            "Tb.Sp (mm)": roi_bone_metrics_patient[2::4],
            "Tb.N (mm^-1)": roi_bone_metrics_patient[3::4]
        })
        
        print(df.to_string(index=False))

        if check_save_output(self.config, "metrics", self.config.save_results.bone_metrics):
            bone_metrics_dir = os.path.join(self.config.experiment_dir, "metrics", "bone_metrics.csv")
            df.to_csv(bone_metrics_dir, index=False)

        
        return




if __name__ == "__main__":
    pass





