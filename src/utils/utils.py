# ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
# ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
# █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
# ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
# ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
# ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#               A Modular Framework for Proximal Femur Analysis
# ==================================================================================
# Filename: utils.py
# Author(s): Marten J. Finck, Niklas C. Koser
# Date: 11.06.2025
# Version: 1.0
# Description: This script implements utility functions.
# ==================================================================================



import numpy as np
import torch
import yaml
import SimpleITK as sitk
import os
from src.utils.config import Config
from datetime import datetime



def load_yaml(path_config_file: str) -> dict:
    """
    Load a yaml file and return the dictionary.

    Parameters:
        path_config_file (str): Path to the yaml file.
    
    Returns:
        config (dict): Dictionary containing the yaml file.
    """
    with open(path_config_file, 'r') as file:
        return yaml.safe_load(file)
    

def print_module_start(module_name: str):
    """
    Prints the name of the module and automatically adds the calculated number of dashes.

    Parameters:
        module_name (str): Name of the module.
    
    Returns:
        None
    """
    print("\n")
    total_number_of_dashes = 82
    number_of_dashes = total_number_of_dashes - len(module_name) - 2
    if number_of_dashes % 2 != 0:
        number_of_dashes -= 1
        dashes = "-" * int(number_of_dashes/2)
        print(f"{dashes} {module_name}  {dashes}")
    else:
        dashes = "-" * int(number_of_dashes/2)
        print(f"{dashes} {module_name} {dashes}")




def load_nifti(path_nifti: str) -> tuple:
    """
    Loads nifti to a numpy array and extracts the meta information.

    Parameters:
        path_nifti (str): Path to the nifti file.

    Returns:
        nifti_data (np.array): Numpy array of the nifti file.
        meta_dict (dict): Dictionary containing the meta information of the nifti file.
    """
    # print(f"---------- Loading nifti file from {path_nifti} ----------")
    print_module_start("Loading Nifti")
    meta_dict = {}
    nifti_img = sitk.ReadImage(path_nifti)
    # Original-Spatial-Infos
    original_spacing = nifti_img.GetSpacing()
    original_size = nifti_img.GetSize()

    # Gewünschtes neues Spacing (z.B. isotrop 1mm)
    new_spacing = (1.5, 1.5, 1.5)

    # Neue Bildgröße berechnen (angepasst ans neue Spacing)
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    # Resampler konfigurieren
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(nifti_img.GetDirection())
    resampler.SetOutputOrigin(nifti_img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)  # Für CT sitkLinear, für Label sitkNearestNeighbor
    resampler.SetDefaultPixelValue(nifti_img.GetPixelIDValue())

    # Resampling durchführen
    resampled_img = resampler.Execute(nifti_img)

    # Optional: speichern
    sitk.WriteImage(resampled_img, path_nifti.replace(".nii.gz","resampled_image.nii.gz"))

    meta_dict['spacing'] = nifti_img.GetSpacing()
    meta_dict['origin'] = nifti_img.GetOrigin()
    meta_dict['direction'] = nifti_img.GetDirection()
    nifti_data = sitk.GetArrayFromImage(nifti_img)
    print(f"Shape of loaded array: {nifti_data.shape}")
    # print("---------- Loading finished ----------\n")
    return nifti_data, meta_dict




def save_nifti(array: np.array, meta_dict: dict, path_saving_dir: str, filename: str) -> None:
    """
    Saves a numpy array as a nifti file. The meta information is used to set the spacing, origin and direction.

    Parameters:
        array (np.array): Numpy array to be saved.
        meta_dict (dict): Dictionary containing the meta information of the nifti file.
        path_saving_dir (str): Path to the saving directory.
        filename (str): Name of the nifti file.

    Returns:
        None
    """
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.SetSpacing(meta_dict['spacing'])
    sitk_image.SetOrigin(meta_dict['origin'])
    sitk_image.SetDirection(meta_dict['direction'])
    sitk.WriteImage(sitk_image, os.path.join(path_saving_dir, filename))
    del sitk_image
    return




def get_device(device_nr: int = 0):
    """
    Returns the device to be used for PyTorch operations. If CUDA is available, it returns the first GPU device.

    Parameters:
        device_nr (int): The index of the GPU device to be used. Default is 0 (first GPU).
    
    Returns:
        device (torch.device): The device to be used for PyTorch operations.
    """
    device = torch.device(f"cuda:{device_nr}")  # Erster GPU (Index 0)

    # Prüfen, ob CUDA verfügbar ist
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA ist nicht verfügbar. CPU wird verwendet.")

    return device


def start_femoralyze():
    """
    Initializes the Femoralyze framework by loading the configuration from a YAML file,
    creating a unique experiment directory based on the current date and time, and printing the configuration.

    Parameters:
        None
    Returns:
        config (Config): The configuration object containing the settings for the experiment.
    """
    config = Config.from_yaml('config.yml')
    print()
    print(config.init_print)

    folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{config.experiment_name}"
    config.experiment_dir = os.path.join(config.results_dir, folder_name).replace("\\","/")

    # Create the experiment directory if it does not exist
    os.makedirs(config.experiment_dir, exist_ok=True)

    print(f"Results will be saved in: {config.experiment_dir}")

    return config



def check_save_output(config, folder_dir_name, config_save_flag):
    """
    Checks if results should be saved based on the configuration and the provided folder directory name.
    If results should be saved, it creates the directory if it does not exist.
    Parameters:
        config (Config): The configuration object containing the settings for the experiment.
        folder_dir_name (str): The name of the folder where results will be saved.
        config_save_flag (bool): A flag indicating whether to save results based on the configuration.
    Returns:
        bool: True if results will be saved, False otherwise.
    """
    if config.save_results.all or config_save_flag:
        folder_dir = os.path.join(config.experiment_dir, folder_dir_name).replace("\\", "/")
        os.makedirs(folder_dir, exist_ok=True)
        return True