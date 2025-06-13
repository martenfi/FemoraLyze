# FemoraLyze ([Paper](https://openreview.net/pdf?id=dgusajTAqF))
````
#               ███████╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗  ██╗   ██╗███████╗███████╗
#               ██╔════╝██╔════╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
#               █████╗  █████╗  ██╔████╔██║██║   ██║██████╔╝███████║██║   ╚████╔╝   ███╔╝ █████╗  
#               ██╔══╝  ██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
#               ██║     ███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║███████╗██║   ███████╗███████╗
#               ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
#                              A Modular Framework for Proximal Femur Analysis
````

## Description
FemoraLyze is a modular framework for standardized analysis of the proximal femur. It enables the quantifiable acquisition of metric parameters of bone architecture and geometry as well as the generation of different segmentation masks for potential downstream applications. The modular structure defines each component solely by its input and output data, ensuring flexible interchangeability. With minor adjustments, the framework can be transferred to other bone structures and thus serves as a basis for standardized, reproducible bone analyses.

## Usage
To use FemoraLyze, download the model weights and templates ([Download](https://cloud.rz.uni-kiel.de/index.php/s/AampAL3eDMFWs4A)) and add the two folders to the repository. 

```
git clone https://github.com/martenfi/FemoraLyze.git
wget -O FemoraLyze.zip "https://cloud.rz.uni-kiel.de/index.php/s/AampAL3eDMFWs4A/download"
unzip FemoraLyze.zip
rm FemoraLyze.zip
```

If you want to use FemoraLyze on your own data, the models may need to be retrained or fine-tuned and the templates and labels adapted. You also need to check whether the axes and angle calculations make sense for the respective application. 

**Note:** FemoraLyze currently works on HRpQCTs of the proximal femur. An in-depth validation with medical input has not been carried out to date, which is why the calculated results should not yet be used for further analyses.


## Results Directory
````
YYYYMMDD_HHMM_Experiment_Name
├── metrics                   
│   ├── axis_measurements.csv  
│   ├── bone_metrics.csv         
│   ├── center_coordinates.csv     
│   ├── cortical_thickness.csv        
│   └── region_volumes.csv
├── pointclouds   
│   ├── pointcloud_surface.pcd           
│   └── pointcloud_volume.pcd
├── regions                   
│   ├── Region_Femur_Head.nii.gz  
│   ├── Region_Femur_Neck.nii.gz         
│   ├── Region_Femur_Shaft.nii.gz          
│   └── Region_Trochanter.nii.gz
├── rois  
│   ├── ROI_Head_128.nii.gz         
│   ├── ROI_Neck_128.nii.gz          
│   └── ROI_Trochanter_128.nii.gz   
└── segmentation_masks  
    ├── bone_mask.nii.gz           
    ├── bone_region_mask.nii.gz           
    └── bone_structure_mask.nii.gz       
````

## Roadmap
**Planned**
- Validation
- Performance Improvements 
    - Processing Time (especially for the Hildebrand algorithm, mask computation)
    - Memory Usage
- Finite Element Analysis (FEA)
- Graphical User Interface (GUI)
- PDF Report Generation

**Optional**
- Train PointNet++ or similar as an alternative for the template matching


## Authors and Acknowledgment
Marten Johannes Finck*, Niklas Christoph Koser*, Jan-Bernd Hövener, Claus-C. Glüer, Sören Pirk

\* Contributed equally

Data was provided by Dr. med. Dr. rer. nat. F. von Brackel and Prof. Dr. med. B. Ondruschka from the University Medical Center Hamburg-Eppendorf (UKE).


## License
Copyright 2025 Visual Computing and Artificial Intelligence, Kiel University, Kiel, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Project Status
The project is still under development. The extracted metrics are not validated yet.
