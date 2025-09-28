# Swin UNETR levee detection model

Buławka, Nazarij, Hector A. Orengo, Felipe Lumbreras Ruiz, Iban Berganzo-Besga and Ekta Gupta. n.d. ‘Leveraging Big Multitemporal Multisource Satellite Data and Artificial Intelligence for the Detection of Complex and Invisible Features - the Case of Extensive Irrigation Mapping’.

*In construction*

## Abstract

The detection of buried or obscured archaeological features remains a central challenge in landscape archaeology, particularly in the irrigated floodplains of Mesopotamia where levees and canals formed the basis of complex agrarian systems. This study presents a deep learning–based approach for the large-scale, semi-automated detection of ancient levees in central Iraq, integrating big multitemporal and multisource satellite datasets with advanced instance segmentation models.
Datasets were assembled from multitemporal Landsat 5, Sentinel-1 SAR, Sentinel-2 multispectral imagery, and the TanDEM-X Edited DSM, combined with vegetation and moisture indices, PCA reductions of seasonal variability, and Multi-Scale Relief Model (MSRM) outputs. Training labels were generated through both threshold-based automatic extraction and detailed manual digitisation. Three architectures: U-Net, Attention U-Net, and Swin UNETR, were evaluated on datasets containing 53, 48, and 36 bands.
Results demonstrate that Swin UNETR consistently outperformed other models, particularly when trained on the 48-band dataset with manually digitised levees. Unlike wide automatic annotations, which produced irregular noisy patches, thin manual annotations yielded clearer, more linear predictions. Post-processing further refined outputs, allowing the model to achieve precision of 0.5118 and recall of 0.5908 at the pixel level. While metric scores remain modest, reflecting the irregularity of the archaeological features, the model successfully predicted levee networks across ~23,600 km², extending from Babylon to Uruk. Comparative analysis with independent palaeochannel reconstructions confirmed that the model identified many of the most prominent irrigation features while avoiding misclassification of modern infrastructure.
The results highlight both the challenges and promise of deep learning in archaeological remote sensing. Automated predictions cannot yet replace interpretative digitisation, but they provide reproducible, standardised, and scalable outputs that can accelerate archaeological mapping and support regional-scale analysis. By leveraging multitemporal, multisource datasets and advanced AI architectures, this study demonstrates a pathway towards reconstructing irrigation systems of different historical periods and landscapes. The approach opens new possibilities for documenting, preserving, and interpreting water-management legacies in some of the world’s most significant ancient landscapes. 


## Workflow

The detailed workflow is described in the publication. A brief description of the procedure can be descibed as follows:

### Train the model
#### Pre-processing
*The code used in this stage is available using following [link](https://github.com/nazarb/2025_levees_DL/tree/main/Dataset)*
- **A** - raster data - Calculate the multisource rasters using published Google Earth Engine [code](https://github.com/nazarb/2025_levees_DL/blob/main/Dataset/Dataset_creation_GEE_code)
- **B** - annotations - Rasterize the levee network created for the purpose of this work available in https://doi.org/10.58132/MGOHM8. Clip and adjust pixels in A and B.
- Create tiles using A and B
- Perform augmentations using [albumentations](https://github.com/albumentations-team/albumentations)
- Create JSON file with structure of the dataset

##### Dataset

The final version of the dataset consist of 48 bands, which are calculated using multispelctral [LANDSAT-5](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2?hl=pl), [Sentinel-2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED?hl=pl), [Sentinel 1](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD?hl=pl) SAR images, and [TanDEM-X 30 EDEM](https://geoservice.dlr.de/web/dataguide/tdm30/).

The code for creating the composite consist of several parts
- mean value of each band of LANDSAT 5 calculated in dry season [[1]](#1)[[2]](#2)
- SMTVI - Seasonal Multi-Temporal Vegetation Indices (SMTVI) using EVI - LANDSAT 5 [[1]](#1)[[2]](#2)[[3]](#3)
- PCA of SMTVI - LANDSAT 5  (this study)
- Standard deviation of SMTVI - LANDSAT 5 (this study)
- NDMI - Normalized Difference Moisture Index (NDMI)- LANDSAT 5 [[4]](#4)
- PCA of NDMI - LANDSAT 5  (this study)
- Standard deviation of NDMI - LANDSAT 5 (this study)
- NDWI - Normalized Difference Water Index (NDWI) - LANDSAT 5  [[5]](#5)[[6]](#6)
- PCA of NDWI - LANDSAT 5  (this study)
- Standard deviation of NDWI - LANDSAT 5 (this study)
- mean value of VV band od Sentinel-1 calculated in dry season [[2]](#2)
- PCA of SMT calculated from VV band od Sentinel-1 (this study)
- Standard deviation of SMT calculated from VV band od Sentinel-1 (this study)
- mean value of VH band od Sentinel-1 calculated in dry season [[2]](#2)
- PCA of SMT calculated from VH band od Sentinel-1 (this study)
- Standard deviation of SMT calculated from VH band od Sentinel-1 (this study)
- mean value of each band of Sentinel-2 calculated in dry season [[2]](#2)[[3]](#3)
- mean value of Misra Soil Brightness Index (MSBI) calculated in dry season - Sentinel-2
- MSRM 1 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 60, x 2 [[7]](#7)[[2]](#2)
- MSRM 2 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 120, x 2 [[7]](#7)[[2]](#2)
- MSRM 3 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 240, x 2 [[7]](#7)[[2]](#2)
- MSRM 4 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 480, x 2 [[7]](#7)[[2]](#2)
- MSRM 5 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 960, x 2 [[7]](#7)[[2]](#2)
- MSRM 6 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 1920, x 2 [[7]](#7)[[2]](#2)
- MSRM 7 - Multi-Scale Relief Model (MSRM) - fmin 0, fmax 3840, x 2 [[7]](#7)[[2]](#2)

#### Train and validate the model 

The research utilizes the Unet, Attention Unet and Swin UNETR developed by [MONAI](https://github.com/Project-MONAI/MONAI)*

The final version of the Swin UNETR model is available [here](https://github.com/nazarb/2025_levees_DL/tree/main/Swin_UNETR)


*The selection phase of the training included separate sets od code for training Unet, Attention Unet and Swin UNETR models*
- Unet [link](https://github.com/nazarb/2025_levees_DL/tree/main/Model)
- Attention Unet [link](https://github.com/nazarb/2025_levees_DL/tree/main/Model)
- Swin UNETR  [link](https://github.com/nazarb/2025_levees_DL/tree/main/Model)

  
#### Predict
*The code used in this stage is available using following [link](https://github.com/nazarb/2025_levees_DL/tree/main/Predict)*

The code use for prediction consist of several steps
- Calculate the multisource rasters using published Google Earth Engine [code](https://github.com/nazarb/2025_levees_DL/blob/main/Dataset/Dataset_creation_GEE_code)
- Run the detection code
	- 	Select raster
	- 	Convert to NPY
	- 	Predict levees
	- 	Convert predictions to TIF and merge them
    
#### Post-processing
*The code used in this stage is available using following [link](https://github.com/nazarb/2025_levees_DL/tree/main/Post_processing)*

Post-processing include:
- filter by size
- closing


## Citation

```bash


@article{bulawkaLeveragingBigMultitemporal01,
	title = {Leveraging big multitemporal multisource satellite data and artificial intelligence for the detection of complex and invisible features - the case of extensive irrigation mapping},
	volume = {},
	doi = {},
	number = {},
	journal = {},
	author = {Buławka, Nazarij and Orengo, Hector A. and Lumbreras Ruiz, Felipe and Berganzo-Besga, Iban and Gupta, Ekta},
	pages = {},
}


```
# References
<a id="1">[1]</a> 
Orengo, Hector A., and Cameron A. Petrie. (2017). 
Large-scale, multi-temporal remote sensing of palaeo-river networks: A case study from Northwest India and its implications for the Indus civilisation.
Remote Sensing, 9(7), 735.
http://www.mdpi.com/2072-4292/9/7/735
https://github.com/horengo/Orengo_Petrie_2017_RS

<a id="2">[2]</a> 
Buławka, Nazarij, and Hector A. Orengo. (2024).
Application of multi-temporal and multisource satellite imagery in the study of irrigated landscapes in arid climates.
Remote Sensing, 16(11), 1997.
https://doi.org/10.3390/rs16111997

<a id="3">[3]</a> 
Garcia-Molsosa, Arnau, Hector A. Orengo, and Cameron A. Petrie. (2023).
Reconstructing long-term settlement histories on complex alluvial floodplains by integrating historical map analysis and remote-sensing: an archaeological analysis of the landscape of the Indus River Basin.
Heritage Science, 11(1), 141.
https://heritagesciencejournal.springeropen.com/articles/10.1186/s40494-023-00985-6

<a id="4">[4]</a> 
Jin, Suming, and Steven A. Sader. (2005). 
Comparison of Time Series Tasseled Cap Wetness and the Normalized Difference Moisture Index in Detecting Forest Disturbances.
Remote Sensing of Environment, 94(3), 364–72. 
https://doi.org/10.1016/j.rse.2004.10.012.

<a id="5">[5]</a> 
Gao, Bo-cai. (1996). 
NDWI—A Normalized Difference Water Index for Remote Sensing of Vegetation Liquid Water from Space.
Remote Sensing of Environment, 58(3), 257–66. 
https://doi.org/10.1016/S0034-4257(96)00067-3.

<a id="6">[6]</a> 
Ji, Lei, Li Zhang, and Bruce Wylie. (2009). 
Analysis of Dynamic Thresholds for the Normalized Difference Water Index. 
Photogrammetric Engineering & Remote Sensing 75(11): 1307–17. 
https://doi.org/10.14358/PERS.75.11.1307.

<a id="7">[7]</a> 
Orengo, Hector A., and Cameron A. Petrie. (2018). 
Multi‐scale Relief Model (MSRM): A New Algorithm for the Visualization of Subtle Topographic Change of Variable Size in Digital Elevation Models/
Earth Surface Processes and Landforms, 43(6), 1361–69. 
https://doi.org/10.1002/esp.4317.
https://github.com/horengo/Orengo_Petrie_2018_MSRM.



Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
