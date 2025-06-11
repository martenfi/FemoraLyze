# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: region_volumes.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script computes the volumes of different bone regions based on
#              the bone region mask.
# ==================================================================================



from src.utils.utils import print_module_start, check_save_output
import pandas as pd 
import numpy as np
import os



class RegionVolumes:
    def __init__(self, config):
        self.config = config
        self.voxel_spacing = config.voxel_spacing_mm
        self.label_meanings = {int(k): v for k, v in self.config.label_meanings._to_dict().items() if k.isdigit()}

    
    def process(self, bone_region_mask: np.ndarray) -> None:
        """
        Computes the volumes of different bone regions in a 3D mask and saves the results to a CSV file.

        Parameters:
            bone_region_mask (numpy.ndarray): 3D numpy array representing the bone region mask.

        Returns:
            None
        """
        print_module_start("Region Volumes")
        voxel_volume = self.voxel_spacing ** 3  # mm^3

        region_counts = {label: (bone_region_mask == label).sum() for label in self.label_meanings.keys()}
        region_volumes_mm3 = {label: count * voxel_volume for label, count in region_counts.items()}
        region_volumes_cm3 = {label: volume / 1000 for label, volume in region_volumes_mm3.items()}

        data = {
            "Region Name": [self.label_meanings[label] for label in self.label_meanings.keys()],
            "Label Number": list(self.label_meanings.keys()),
            "Voxel Count": [region_counts[label] for label in self.label_meanings.keys()],
            "Volume (mm^3)": [region_volumes_mm3[label] for label in self.label_meanings.keys()],
            "Volume (cm^3)": [region_volumes_cm3[label] for label in self.label_meanings.keys()]
        }
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        if check_save_output(self.config, "metrics", self.config.save_results.region_volumes):
            region_volumes_dir = os.path.join(self.config.experiment_dir, "metrics", "region_volumes.csv")
            df.to_csv(region_volumes_dir, index=False)
        

        return 




if __name__ == "__main__":
    pass