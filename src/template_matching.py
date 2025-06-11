# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: template_matching.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script performs template matching using a 3-stage-template 
#              matching process on surface point clouds.
# ==================================================================================



import numpy as np
import open3d as o3d
from skimage import measure
import copy
import os
from src.utils.utils import load_nifti, print_module_start, check_save_output




class TemplateMatching:
    def __init__(self,config):
        self.config = config
        self.label_colors = {int(k): v for k, v in self.config.label_colors._to_dict().items() if k.isdigit()}
        self.label_meanings = {int(k): v for k, v in self.config.label_meanings._to_dict().items() if k.isdigit()}
        self.number_of_points = self.config.template_matching.number_of_points
        self.path_templates = self.config.template_matching.path_templates

        #Ransac parameters
        self.radius_feature = self.config.template_matching.ransac.radius_feature
        self.max_nn = self.config.template_matching.ransac.max_nn
        self.distance_threshold_ransac = self.config.template_matching.ransac.distance_threshold_ransac
        self.mutual_filter = self.config.template_matching.ransac.mutual_filter
        self.scaling_ransac = self.config.template_matching.ransac.scaling
        self.ransac_n = self.config.template_matching.ransac.ransac_n
        self.similarity_threshold = self.config.template_matching.ransac.similarity_threshold
        self.max_iteration_ransac = self.config.template_matching.ransac.max_iteration
        self.confidence = self.config.template_matching.ransac.confidence

        #ICP parameters
        self.scaling_icp = self.config.template_matching.icp.scaling
        self.distance_threshold_icp = self.config.template_matching.icp.distance_threshold_icp
        self.relative_fitness = self.config.template_matching.icp.relative_fitness
        self.relative_rmse = self.config.template_matching.icp.relative_rmse
        self.max_iteration_icp = self.config.template_matching.icp.max_iteration


    def __get_pointcloud(self, bone_mask: np.array, number_of_points: int) -> o3d.geometry.PointCloud:
        """
        Generate a pointcloud from bone mask by using marching cubes algorithm and sample points uniformly on the surface.
        
        Parameters:
            bone_mask (np.array): Numpy array of the bone mask. The bone mask is expected to be a binary mask with 1s for bone and 0s for background.
            number_of_points (int): Number of points to sample from the surface of the mesh.
        Returns:
            pcd (o3d.geometry.PointCloud): Open3D pointcloud object with sampled points from the surface of the mesh.
        """
        verts, faces, normals, values = measure.marching_cubes(bone_mask, level=0.5)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces[:, ::-1]) # Reverse the order of the vertices to ensure correct orientation
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
        return pcd


    def __visualize_pcds(self, pcds: list, camera_params: dict) -> None:
        """
        Visualize multiple pointclouds in a 3D plot.

        Parameters:
            pcds (list): List of Open3D pointcloud objects.
            camera_params (dict): Dictionary containing the camera parameters for the visualization.
        
        Returns:
            None
        """
        front = camera_params["front"]
        lookat = camera_params["lookat"]
        up = camera_params["up"]
        zoom = camera_params["zoom"]
        o3d.visualization.draw_geometries(pcds, window_name="Pointcloud(s)", point_show_normal=False, front=front, lookat=lookat, up=up, zoom=zoom)


    def __draw_registration_result(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, camera_params: dict) -> None:
        """
        Visualize the registration result of two pointclouds.

        Parameters:
            source (o3d.geometry.PointCloud): Source pointcloud.
            target (o3d.geometry.PointCloud): Target pointcloud.
            camera_params (dict): Dictionary containing the camera parameters for the visualization.
        
        Returns:
            None
        """
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        target_temp.paint_uniform_color([0.5, 0.5, 0.5])
        self.__visualize_pcds([source_temp, target_temp], camera_params)


    def __three_stage_pcd_registration(self, pcd_source: o3d.geometry.PointCloud, pcd_target: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Registration using a three-stage approach: First the volume centers are aligned, then a coarse registration with RANSAC is performed and finally a fine registration with ICP.

        Parameters:
            pcd_source (o3d.geometry.PointCloud): Source pointcloud.
            pcd_target (o3d.geometry.PointCloud): Target pointcloud.

        Returns:
            source (o3d.geometry.PointCloud): Registered source pointcloud.
        """

        # camera_params = {
        #     'front': [0.22948383329683952, 0.83847752862109948, 0.49425965266531907],
        #     'lookat': [883.0, 335.0, 536.5],
        #     'up': [-0.97304793641690435, 0.20947797388795364, 0.096419354336158392],
        #     'zoom': 0.67999999999999994
        # }
        # self.draw_registration_result(pcd_source, pcd_target, camera_params)

        source = copy.deepcopy(pcd_source)
        target = copy.deepcopy(pcd_target)
      
        # Centroid mapping
        print("Stage 1: Centroid mapping")
        center_source = pcd_source.get_center()
        center_target = pcd_target.get_center()
        translation = center_target - center_source
        source.translate(translation)
        # self.draw_registration_result(source, target, camera_params)

        # Coarse registration with RANSAC
        print("Stage 2: Coarse registration with RANSAC")
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_feature, max_nn=self.max_nn))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_feature, max_nn=self.max_nn))
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, self.mutual_filter, self.distance_threshold_ransac,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(self.scaling_ransac), self.ransac_n, 
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(self.similarity_threshold), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.distance_threshold_ransac)], 
            o3d.pipelines.registration.RANSACConvergenceCriteria(self.max_iteration_ransac, self.confidence))
        initial_transformation = result_ransac.transformation
        source.transform(initial_transformation)
        # self.draw_registration_result(source, target, camera_params)

        # Fine registration with ICP
        print("Stage 3: Fine registration with ICP")
        scaling = self.scaling_icp
        distance_threshold_icp = self.distance_threshold_icp
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold_icp, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=scaling),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = self.relative_fitness , relative_rmse = self.relative_rmse, max_iteration = self.max_iteration_icp))
        icp_transformation = result_icp.transformation
        source.transform(icp_transformation)
        # self.draw_registration_result(source, target, camera_params)

        return source


    def __push_labels(self, registered_source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Push the labels from the registeresd source pointcloud to the target pointcloud by finding the nearest neighbors.

        Parameters:
            registered_source (o3d.geometry.PointCloud): Registered source pointcloud.
            target (o3d.geometry.PointCloud): Target pointcloud.

        Returns:
            labeled_target (o3d.geometry.PointCloud): Target pointcloud with the labels from the registered source pointcloud.
        """
        labeled_target = copy.deepcopy(target)
        source_colors = np.asarray(registered_source.colors)
        target_points = np.asarray(labeled_target.points)
        target_colors = []

        # Create a KDTree for the source point cloud
        source_tree = o3d.geometry.KDTreeFlann(registered_source)
        for target_point in target_points:
            # Find the closest point in the source point cloud
            _, idx, _ = source_tree.search_knn_vector_3d(target_point, 1)
            # Assign the color of the closest source point
            target_colors.append(source_colors[idx[0]])
        # Update target point cloud colors
        labeled_target.colors = o3d.utility.Vector3dVector(target_colors)

        return labeled_target

    
    def process(self, bone_mask: np.array) -> o3d.geometry.PointCloud:
        """
        Perform template matching on the given bone mask by using a three-stage-template matching process.

        Parameters:
            bone_mask (np.array): Numpy array of the bone mask. The bone mask is expected to be a binary mask with 1s for bone and 0s for background.
        
        Returns:
            pcd_target_labels (o3d.geometry.PointCloud): Open3D pointcloud object with the labels from the template matching process.
        """
        print_module_start("Template Matching")
        number_of_points = self.number_of_points
        print(f"Number of sampled surface points: {number_of_points}")

        # Paths
        # path_templates = r"data\templates"
        path_templates = self.path_templates
        path_templates_bone_masks = os.path.join(path_templates,"bone_mask")
        path_templates_pcds = os.path.join(path_templates,"pcd")

        # Template voxel counts
        if os.path.exists(os.path.join(path_templates,"template_voxel_counts.npy")):
            template_voxel_count = np.load(os.path.join(path_templates,"template_voxel_counts.npy")).tolist()
        else:
            template_voxel_count = []
            for path_template in [f for f in os.listdir(path_templates_bone_masks) if f.endswith(".nii.gz")]:
                template_bone_mask, _ = load_nifti(os.path.join(path_templates_bone_masks, path_template))
                template_bone_mask_voxel_count = int(np.sum(template_bone_mask))
                template_voxel_count.append(template_bone_mask_voxel_count)
            np.save(os.path.join(path_templates,"template_voxel_counts.npy"), template_voxel_count)

        # Get voxelcount of the bone mask
        voxel_target = bone_mask.copy()
        pcd_target = self.__get_pointcloud(voxel_target, number_of_points)
        target_voxel_count = int(np.sum(voxel_target))

        # Selcet the template with the most similar voxel count
        template_index = np.argmin([abs(target_voxel_count - count) for count in template_voxel_count])
        template_pcd = o3d.io.read_point_cloud(os.path.join(path_templates_pcds, f"template_{template_index}.pcd"))

        # Template matching and label assignment
        pcd_source_registered = self.__three_stage_pcd_registration(template_pcd, pcd_target)
        pcd_target_labels = self.__push_labels(pcd_source_registered, pcd_target)

        if check_save_output(self.config, "pointclouds", self.config.save_results.pcd_surface):
            pcd_folder_path = os.path.join(self.config.experiment_dir, "pointclouds", "pointcloud_surface.pcd")
            o3d.io.write_point_cloud(pcd_folder_path, pcd_target_labels)

        return pcd_target_labels






if __name__ == "__main__":
    pass
    