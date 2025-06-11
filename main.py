# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: main.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script connects all the modules and runs the entire pipeline.
# ==================================================================================



from src.utils.utils import load_nifti, start_femoralyze
from src.nnUnet import MaskProcessor
from src.template_matching import TemplateMatching
from src.bone_region_mask import BoneRegionMask
from src.region_volumes import RegionVolumes
from src.center_points import CenterPoints
from src.axis_measurements import AxisMeasurements
from src.roi_extraction import RoiExtraction
from src.bone_metrics import BoneMetrics
from src.region_extraction import RegionExtraction
from src.cortical_thickness import CorticalThickness



def main():


    config = start_femoralyze()

    # Load CT file
    image_arr, meta_dict = load_nifti(config.input_dir)

    # Compute nnUnet Segementation
    bone_mask, bone_structure_mask, bone_cortical_mask = MaskProcessor(config).process(image_arr, meta_dict)
    
    # Template Matching
    pcd_labeled = TemplateMatching(config).process(bone_mask)
    
    # Bone Region Mask
    bone_region_mask = BoneRegionMask(config).process(pcd_labeled, bone_mask, meta_dict)

    
    # Compute Region Volumes
    RegionVolumes(config).process(bone_region_mask)
    
    # Center Points
    center_points = CenterPoints(config).process(pcd_labeled)
    
    # Axis Measurements
    AxisMeasurements(config).process(center_points)
    
    # Extract ROIs
    rois = RoiExtraction(config).process(bone_structure_mask, center_points, meta_dict)
    
    # Compute Bone Metrics
    BoneMetrics(config).process(rois)
    
    # Extract Regions (of cortical mask)
    bone_regions = RegionExtraction(config).process(bone_region_mask, bone_cortical_mask, meta_dict)
    
    # Compute Cortical Thickness (exclude due to time constraints)
    # cortical_thickness = CorticalThickness(config).process(bone_regions[3])

if __name__ == "__main__":
    main()
