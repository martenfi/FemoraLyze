# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: nnUnet.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script defines the Inference of the nnUNet model.
# ==================================================================================



import os
import tempfile
from pathlib import Path
import monai
import numpy as np
from src.utils.utils import save_nifti, get_device, print_module_start, check_save_output


class nnUNetInference:
    def __init__(self,model_path, checkpoint_name, tile_step=0.8,  fold_nr=0, model_spacing=None, device_nr=2):
        self.tile_step = tile_step
        self.checkpoint_name = checkpoint_name
        self.model_path = model_path
        self.fold_nr = fold_nr
        self.device = get_device(device_nr=device_nr)
        self.model_spacing = model_spacing

        self.set_paths()

        # init modules
        self.__init_predictor()
        self.__load_model()

    def set_paths(self) -> None:
        """
        Set the environment variables for nnUNet paths.

        Parameters:
            None
            
        Returns:
            None
        """
        dummy_dir = Path(tempfile.mkdtemp())
        os.environ.update({
            'nnUNet_raw': str(dummy_dir / "raw"),
            'nnUNet_preprocessed': str(dummy_dir / "preprocessed"),
            'nnUNet_results': self.model_path # HIER MUSST DU DEINEN MODELLPFAD EINTRAGEN
        })

    def __init_predictor(self) -> None:
        """
        Initialize the nnUNet predictor with the specified parameters.

        Parameters:
            None
        
        Returns:
            None
        """
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        self.predictor = nnUNetPredictor(
            tile_step_size=self.tile_step,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device= True,
            device=self.device,
            verbose=False,
        )

    def __load_model(self) -> None:
        """
        Load the trained nnUNet model from the specified path.

        Parameters:
            None

        Returns:
            None
        """
        if self.predictor is not None:
            self.predictor.initialize_from_trained_model_folder(
                model_training_output_dir = self.model_path,
                use_folds = (self.fold_nr,),
                checkpoint_name= self.checkpoint_name
            )

    def expand_axis(self, image_arr: np.ndarray) -> np.ndarray:
        """
        Expand the dimensions of the input image array to match the expected input shape for the nnUNet model.

        Parameters:
            image_arr (np.ndarray): The input image array to be expanded.
        
        Returns:
            np.ndarray: The expanded image array with an additional dimension at the front.
        """
        return np.expand_dims(image_arr, axis=0)

    def predict(self, image_arr: np.ndarray,spacing=None) -> np.ndarray:
        """
        Predict the segmentation of the input image array using the nnUNet model.

        Parameters:
            image_arr (np.ndarray): The input image array to be segmented.
            spacing (tuple, optional): The spacing of the input image. If not provided, uses the model's default spacing.

        Returns:
            np.ndarray: The predicted segmentation of the input image array.
        """
        segmentation = self.predictor.predict_single_npy_array(
            input_image = image_arr,
            image_properties = {
                'spacing': self.model_spacing,
                #'dtype': 'float32',
            }  if self.model_spacing else  {
                'spacing': spacing,
                #'dtype': 'float32',
            } ,
            segmentation_previous_stage = None,
            output_file_truncated = None,
            save_or_return_probabilities = False,
        )

        return segmentation



class MaskProcessor:
    def __init__(self, config):
        self.config = config

    def process_mask(self, mask_config, input_image, meta_dict, output_suffix):
    # def process_mask(self, mask_config: Config, input_image: np.ndarray, meta_dict: dict, output_suffix: str) -> np.ndarray:
        """
        Process a mask using the nnUNetInference class.

        Parameters:
            mask_config (Config): Configuration for the mask processing.
            input_image (np.ndarray): The input image to be processed.
            meta_dict (dict): Metadata dictionary containing information about the input image.
            output_suffix (str): Suffix for the output file name.
        
        Returns:
            np.ndarray: The processed mask.
        """
        print(f"Predicting {output_suffix}")

        # inferencer = nnUNetInference(**mask_config.params._to_dict())
        params_dict = mask_config.params._to_dict()
        params_dict.pop("data", None)  # Remove 'data' key if it exists
        inferencer = nnUNetInference(**params_dict)
        mask = inferencer.predict(inferencer.expand_axis(input_image), meta_dict['spacing'])

        return mask

    def get_masked_image(self) -> np.ndarray:
        """
        Gets the masked image array.

        Parameters:
            None

        Returns:
            np.ndarray: The masked image array.

        """
        return self.masked_image_arr

    def preprocess_image(self, image_arr: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image array by clipping intensity percentiles and scaling intensity.

        Parameters:
            image_arr (np.ndarray): The input image array to be preprocessed.

        Returns:
            np.ndarray: The preprocessed image array.
        """
        transforms = monai.transforms.Compose([
            monai.transforms.ClipIntensityPercentiles(lower=0.3, upper=99.7),
            monai.transforms.ScaleIntensity(minv=0, maxv=1),
        ])

        # Apply the transformations to the image
        image_arr = transforms(image_arr)
        return image_arr

    def process(self, image_arr=None, meta_dict=None) -> tuple:
        """
        Generate the bone mask, bone structure mask, and bone cortical mask from the input image array.

        Parameters:
            image_arr (np.ndarray): The input image array to be processed.
            meta_dict (dict): Metadata dictionary containing information about the input image.
        
        Returns:
            tuple: A tuple containing the bone mask, bone structure mask, and bone cortical mask.
        """
        # print(f"Processing {self.config.input_dir}")
        print_module_start("nnUnet Segmentation")
        #image_arr = self.preprocess_image(image_arr)


        # Bone Mask
        bone_mask = self.process_mask(self.config.nnunet_bone_mask, image_arr, meta_dict, "bone_mask")
        if check_save_output(self.config, "segmentation_masks", self.config.save_results.bone_mask):
            seg_dir_path = os.path.join(self.config.experiment_dir, "segmentation_masks")
            save_nifti(bone_mask, meta_dict, seg_dir_path,"bone_mask.nii.gz")



        masked_image_arr = self.preprocess_image(image_arr)
        self.masked_image_arr = masked_image_arr*bone_mask #np.multiply(image_arr, bone_mask)



        # Bone Structure Mask
        bone_structure_mask = self.process_mask(self.config.nnunet_bone_structure_mask, self.masked_image_arr, meta_dict,
                          "bone_structure_mask")
        if check_save_output(self.config, "segmentation_masks", self.config.save_results.bone_structure_mask):
            seg_dir_path = os.path.join(self.config.experiment_dir, "segmentation_masks")
            save_nifti(bone_structure_mask, meta_dict, seg_dir_path,"bone_structure_mask.nii.gz")


        # Bone Cortical Mask
        bone_cortical_mask = self.process_mask(self.config.nnunet_cortical_mask, self.masked_image_arr, meta_dict, "cortical_mask")
        if check_save_output(self.config, "segmentation_masks", self.config.save_results.bone_cortical_mask):
            seg_dir_path = os.path.join(self.config.experiment_dir, "segmentation_masks")
            save_nifti(bone_cortical_mask, meta_dict, seg_dir_path,"cortical_mask.nii.gz")

            

        return bone_mask, bone_structure_mask, bone_cortical_mask
    

if __name__ == "__main__":
    pass