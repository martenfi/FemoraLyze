# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: bone_region_mask.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script computes the bone region mask based on the point cloud.
# ==================================================================================



import numpy as np
import open3d as o3d
import os
from src.utils.utils import save_nifti, check_save_output
from scipy.spatial import Delaunay, cKDTree
from cupyx.scipy.ndimage import binary_closing
import cupy as cp
from src.utils.utils import print_module_start
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()




class BoneRegionMask:
    def __init__(self, config):
        self.config = config
        self.label_colors = {int(k): v for k, v in self.config.label_colors._to_dict().items() if k.isdigit()}
        self.factor_new_points = self.config.bone_region_mask.factor_new_points
        self.pp_radius = self.config.bone_region_mask.pp_radius
        self.pp_iterations = self.config.bone_region_mask.pp_iterations
        self.pp_brute_force = self.config.bone_region_mask.pp_brute_force



    def __select_subpointcloud(self, pcl: o3d.geometry.PointCloud, label: int, label_colors: dict) -> o3d.geometry.PointCloud:
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
    


    def __compute_hulls(self, pcl_head: o3d.geometry.PointCloud, pcl_neck: o3d.geometry.PointCloud, pcl_trochanter: o3d.geometry.PointCloud, pcl_shaft: o3d.geometry.PointCloud) -> tuple:
        """
        Compute convex hulls for the given point clouds and perform boolean operations to remove intersections.

        Parameters:
            pcl_head (o3d.geometry.PointCloud): Point cloud for the head region.
            pcl_neck (o3d.geometry.PointCloud): Point cloud for the neck region.
            pcl_trochanter (o3d.geometry.PointCloud): Point cloud for the trochanter region.
            pcl_shaft (o3d.geometry.PointCloud): Point cloud for the shaft region.

        Returns:
            hull_head_delaunay (scipy.spatial.Delaunay): Delaunay triangulation for the head region.
            hull_neck_delaunay (scipy.spatial.Delaunay): Delaunay triangulation for the neck region.
            hull_trochanter_delaunay (scipy.spatial.Delaunay): Delaunay triangulation for the trochanter region.
            hull_shaft_delaunay (scipy.spatial.Delaunay): Delaunay triangulation for the shaft region.
        """

        hull_head, _ = pcl_head.compute_convex_hull()
        hull_neck, _ = pcl_neck.compute_convex_hull()
        hull_trochanter, _ = pcl_trochanter.compute_convex_hull()
        hull_shaft, _ = pcl_shaft.compute_convex_hull()

        hull_head = o3d.t.geometry.TriangleMesh.from_legacy(hull_head)
        hull_neck = o3d.t.geometry.TriangleMesh.from_legacy(hull_neck)
        hull_trochanter = o3d.t.geometry.TriangleMesh.from_legacy(hull_trochanter)
        hull_shaft = o3d.t.geometry.TriangleMesh.from_legacy(hull_shaft)
        
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

        intersection_neck_head = hull_neck.boolean_intersection(hull_head).to_legacy()
        intersection_trochanter_neck = hull_trochanter.boolean_intersection(hull_neck).to_legacy()
        intersection_shaft_trochanter = hull_shaft.boolean_intersection(hull_trochanter).to_legacy()

        if len(intersection_neck_head.vertices) > 0 and len(intersection_neck_head.triangles) > 0:
            # print("Intersection found between head and neck hulls.")
            hull_neck = hull_neck.boolean_difference(hull_head)
        if len(intersection_trochanter_neck.vertices) > 0 and len(intersection_trochanter_neck.triangles) > 0:
            # print("Intersection found between trochanter and neck hulls.")
            hull_trochanter = hull_trochanter.boolean_difference(hull_neck)
        if len(intersection_shaft_trochanter.vertices) > 0 and len(intersection_shaft_trochanter.triangles) > 0:
            # print("Intersection found between shaft and trochanter hulls.")
            hull_shaft = hull_shaft.boolean_difference(hull_trochanter)

        hull_head = hull_head.to_legacy()
        hull_neck = hull_neck.to_legacy()
        hull_trochanter = hull_trochanter.to_legacy()
        hull_shaft = hull_shaft.to_legacy()

        hull_head_delaunay = Delaunay(np.asarray(pcl_head.points))
        hull_neck_delaunay = Delaunay(np.asarray(pcl_neck.points))
        hull_trochanter_delaunay = Delaunay(np.asarray(pcl_trochanter.points))
        hull_shaft_delaunay = Delaunay(np.asarray(pcl_shaft.points))

        return hull_head_delaunay, hull_neck_delaunay, hull_trochanter_delaunay, hull_shaft_delaunay



    def __populate_hull(self, sub_pcl: o3d.geometry.PointCloud, delaunay: Delaunay, delaunay_exclude: Delaunay, bone_mask: np.ndarray, label: int, label_colors: dict) -> o3d.geometry.PointCloud:
        """
        Populate the convex hull of a point cloud with new points.

        Parameters:
            sub_pcl (o3d.geometry.PointCloud): Subpoint cloud to be populated.
            delaunay (scipy.spatial.Delaunay): Delaunay triangulation of the subpoint cloud.
            delaunay_exclude (scipy.spatial.Delaunay): Delaunay triangulation to exclude points from.
            bone_mask (numpy.ndarray): 3D binary mask of the bone region.
            label (int): Label for the point cloud.
            label_colors (dict): Dictionary containing the label colors.

        Returns:
            combined_pcl (o3d.geometry.PointCloud): Combined point cloud with new points added.
        """

        num_new_points = len(sub_pcl.points)*self.factor_new_points
        min_bound = np.min(delaunay.points, axis=0)
        max_bound = np.max(delaunay.points, axis=0)

        # Function to check if a point is inside the convex hull
        if delaunay_exclude != None:
            def is_valid_point(point, delaunay, delaunay_exclude):
                return delaunay.find_simplex(point) >= 0 and delaunay_exclude.find_simplex(point) < 0
        else:
            def is_valid_point(point, delaunay, delaunay_exclude):
                return delaunay.find_simplex(point) >= 0

        # Generate points within the convex hull
        new_points_within_hull = []
        while len(new_points_within_hull) < num_new_points:
            point = np.random.uniform(low=min_bound, high=max_bound)
            if is_valid_point(point, delaunay, delaunay_exclude):
                new_points_within_hull.append(point)
        
        new_points_within_hull = np.array(new_points_within_hull)

        # Create a new point cloud for the new points
        new_pcl = o3d.geometry.PointCloud()
        new_pcl.points = o3d.utility.Vector3dVector(new_points_within_hull)

        # Combine the original point cloud and the new point cloud
        combined_pcl = sub_pcl + new_pcl
        label_color = label_colors[label]
        combined_pcl.colors = o3d.utility.Vector3dVector(np.tile(label_color, (len(combined_pcl.points), 1)))

        # Remove points that are outside of the bone_mask array
        points = np.asarray(combined_pcl.points)
        colors = np.asarray(combined_pcl.colors)
        valid_points = []
        valid_colors = []

        for point, color in zip(points, colors):
            x, y, z = point
            x, y, z = int(x), int(y), int(z)
            if 0 <= x < bone_mask.shape[0] and 0 <= y < bone_mask.shape[1] and 0 <= z < bone_mask.shape[2]:
                if bone_mask[x, y, z] != 0:
                    valid_points.append(point)
                    valid_colors.append(color)

        combined_pcl.points = o3d.utility.Vector3dVector(valid_points)
        combined_pcl.colors = o3d.utility.Vector3dVector(valid_colors)

        return combined_pcl



    def __populate_pcl(self, pcl: o3d.geometry.PointCloud, label_colors: dict, bone_mask: np.ndarray) -> tuple:
        """
        Populate all pointclouds with new points based on the convex hulls of the subpoint clouds.

        Parameters:
            pcl (o3d.geometry.PointCloud): The original point cloud containing all regions.
            label_colors (dict): Dictionary containing the label colors.
            bone_mask (numpy.ndarray): 3D binary mask of the bone region.
        
        Returns:
            pcl_head_filled (o3d.geometry.PointCloud): Point cloud for the head region with new points.
            pcl_neck_filled (o3d.geometry.PointCloud): Point cloud for the neck region with new points.
            pcl_trochanter_filled (o3d.geometry.PointCloud): Point cloud for the trochanter region with new points.
            pcl_shaft_filled (o3d.geometry.PointCloud): Point cloud for the shaft region with new points. 
        """
        pcl_head = self.__select_subpointcloud(pcl, 2, label_colors)
        pcl_neck = self.__select_subpointcloud(pcl, 4, label_colors)
        pcl_trochanter = self.__select_subpointcloud(pcl, 1, label_colors)
        pcl_shaft = self.__select_subpointcloud(pcl, 3, label_colors)
        hull_head_delaunay, hull_neck_delaunay, hull_trochanter_delaunay, hull_shaft_delaunay = self.__compute_hulls(pcl_head, pcl_neck, pcl_trochanter, pcl_shaft)

        pcl_head_filled = self.__populate_hull(pcl_head, hull_head_delaunay, None, bone_mask, 2, label_colors)
        pcl_neck_filled = self.__populate_hull(pcl_neck, hull_neck_delaunay,hull_head_delaunay, bone_mask, 4, label_colors)
        pcl_trochanter_filled = self.__populate_hull(pcl_trochanter, hull_trochanter_delaunay, hull_neck_delaunay, bone_mask, 1, label_colors)
        pcl_shaft_filled = self.__populate_hull(pcl_shaft, hull_shaft_delaunay, hull_trochanter_delaunay, bone_mask, 3, label_colors)
    
        return pcl_head_filled, pcl_neck_filled, pcl_trochanter_filled, pcl_shaft_filled



    def __assign_labels_to_voxels(self, bone_mask: np.ndarray, pcl: o3d.geometry.PointCloud, label_colors: dict) -> np.ndarray:
        """
        Assign labels (0-4) to each voxel in the bone mask based on the closest point in the point cloud.

        Parameters:
            bone_mask (numpy.ndarray): A binary 3D array where 1 represents bone and 0 represents background.
            pcl (open3d.geometry.PointCloud): An Open3D PointCloud object with points and colors.
            label_colors (dict): A dictionary mapping integer labels (0-4) to RGB color values.

        Returns:
            numpy.ndarray: A 3D array with the same shape as bone_mask containing integer labels (0-4).
        """

        # Ensure the point cloud has points and colors
        if not pcl.has_points() or not pcl.has_colors():
            raise ValueError("Point cloud must have both points and colors.")

        # Extract coordinates and colors from point cloud
        coordinates = np.asarray(pcl.points)  # Shape (N, 3)
        colors = np.asarray(pcl.colors)       # Shape (N, 3)

        # Convert colors to corresponding integer labels
        # color_labels = np.argmax(np.all(colors[:, None, :] == label_colors, axis=2), axis=1)
        label_array = np.array([label_colors[k] for k in sorted(label_colors.keys())])
        label_keys = np.array(sorted(label_colors.keys()))
        color_labels = label_keys[np.argmax(np.all(colors[:, None, :] == label_array, axis=2), axis=1)]

        # Get the voxel coordinates where bone_mask is 1
        bone_voxels = np.argwhere(bone_mask == 1)  # Shape (M, 3)

        # print("Creating cKDTree...")
        tree = cKDTree(coordinates)  # Faster than sklearn KDTree

        # print("Querying cKDTree...")
        _, indices = tree.query(bone_voxels, k=1, workers=-1)  # Batch query

        # Assign the corresponding integer labels to the voxels
        labeled_voxels = np.zeros(bone_mask.shape, dtype=np.int32)
        labeled_voxels[bone_voxels[:, 0], bone_voxels[:, 1], bone_voxels[:, 2]] = color_labels[indices]

        return labeled_voxels



    def __postprocess(self, labeled_voxels: np.ndarray, r: int, i: int) -> np.ndarray:
        """
        Postprocess the labeled voxels using binary closing to refine the segmentation.

        Parameters:
            labeled_voxels (numpy.ndarray): A 3D array with integer labels (0-4) representing the bone regions.
            r (int): Radius for the spherical structuring element used in binary closing.
            i (int): Number of iterations for the binary closing operation.
        
        Returns:
            numpy.ndarray: A 3D array with refined labels after postprocessing.
        """
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        labeled_voxels_cp = cp.array(labeled_voxels)

        #TODO: Use dict to map labels to regions
        head = labeled_voxels_cp == 2
        neck = labeled_voxels_cp == 4
        trochanter = labeled_voxels_cp == 1
        shaft = labeled_voxels_cp == 3

        binary_head_neck = head | neck
        binary_neck_trochanter = neck | trochanter | head
        binary_trochanter_shaft = trochanter | shaft | neck

        def create_spherical_structuring_element(radius):
            # Create a grid of coordinates
            x, y, z = cp.meshgrid(
                cp.arange(-radius, radius + 1),
                cp.arange(-radius, radius + 1),
                cp.arange(-radius, radius + 1),
                indexing='ij'
            )
            
            # Calculate the distance from the center
            distance = cp.sqrt(x**2 + y**2 + z**2)
            
            # Create a binary mask representing the sphere
            structuring_element = distance <= radius
            
            return structuring_element

        struct = create_spherical_structuring_element(r)

        head_closed = binary_closing(head, structure=struct, iterations=i, mask=binary_head_neck, brute_force=self.pp_brute_force)
        neck_closed = binary_closing(neck, structure=struct, iterations=i, mask=binary_neck_trochanter, brute_force=self.pp_brute_force)
        trochanter_closed = binary_closing(trochanter, structure=struct, iterations=i, mask=binary_trochanter_shaft, brute_force=self.pp_brute_force)

        processed_voxels = labeled_voxels_cp.copy()
        processed_voxels = cp.where(trochanter_closed, 1, processed_voxels)
        processed_voxels = cp.where(neck_closed, 4, processed_voxels)
        processed_voxels = cp.where(head_closed, 2, processed_voxels)

        return cp.asnumpy(processed_voxels)  # Convert to NumPy only once



    def process(self, labeled_pointcloud: o3d.geometry.PointCloud, bone_mask: np.ndarray, meta_dict: dict) -> np.ndarray:
        """
        Generate a bone region mask from the labeled point cloud and the bone mask.

        Parameters:
            labeled_pointcloud (o3d.geometry.PointCloud): The labeled point cloud containing bone regions.
            bone_mask (numpy.ndarray): A binary 3D array where 1 represents bone and 0 represents background.
            meta_dict (dict): Metadata dictionary containing information about the bone mask.
        
        Returns:
            numpy.ndarray: A 3D array with integer labels (0-4) representing the bone regions.
        """
        print_module_start("Bone Region Mask")

        # Fill Surface PCL with Points
        pcl_head_filled, pcl_neck_filled, pcl_trochanter_filled, pcl_shaft_filled = self.__populate_pcl(labeled_pointcloud, self.label_colors, bone_mask)
        pcl_new = pcl_head_filled + pcl_neck_filled + pcl_trochanter_filled + pcl_shaft_filled

        if check_save_output(self.config, "pointclouds", self.config.save_results.pcd_volume):
            pcd_dir = os.path.join(self.config.experiment_dir, "pointclouds", "pointcloud_volume.pcd")
            o3d.io.write_point_cloud(pcd_dir, pcl_new)

            
        # Assign Labels to Voxels
        labeled_voxels = self.__assign_labels_to_voxels(bone_mask, pcl_new, self.label_colors)

        # Postprocess Labeled Voxels
        labeled_voxels = labeled_voxels.astype(np.int8)
        processed_voxels = self.__postprocess(labeled_voxels, self.pp_radius, self.pp_iterations)
        processed_voxels = processed_voxels.astype(np.int16)

        if check_save_output(self.config, "segmentation_masks", self.config.save_results.bone_region_mask):
            seg_mask_dir = os.path.join(self.config.experiment_dir, "segmentation_masks")
            save_nifti(processed_voxels, meta_dict, seg_mask_dir, "bone_region_mask.nii.gz")

        return processed_voxels





if __name__ == "__main__":
    pass


