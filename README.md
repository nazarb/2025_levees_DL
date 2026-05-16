# Leveraging Big Multitemporal Multisource Satellite Data and Artificial Intelligence for the Detection of Complex and Invisible Features - the Case of Extensive Irrigation Mapping: Swin UNETR levee detection model

Buławka, N., Orengo, H. A., Lumbreras Ruiz, F., Berganzo-Besga, I., & Gupta, E. (2026). Leveraging big multitemporal multisource satellite data and artificial intelligence for the detection of complex and invisible features—The case of extensive irrigation mapping. Archaeological Prospection, 2158411.



*Accepted for publication*

## Abstract

The detection of buried or obscured archaeological features remains a central challenge in landscape archaeology, particularly in the irrigated floodplains of Mesopotamia where levees and canals formed the basis of complex agrarian systems. This study presents a deep learning–based approach for the large-scale, automated detection of ancient levees in central Iraq, integrating big multitemporal and multisource satellite datasets with advanced instance segmentation models.
Datasets were assembled from multitemporal Landsat 5, Sentinel-1 SAR, Sentinel-2 multispectral imagery, and the TanDEM-X Edited DSM, combined with vegetation and moisture indices, PCA reductions of seasonal variability, and Multi-Scale Relief Model (MSRM) outputs. Training labels were generated through both threshold-based automatic extraction and detailed manual digitisation. Three architectures: U-Net, Attention U-Net, and Swin UNETR, were evaluated on datasets containing 53, 48, and 36 bands.
Results demonstrate that Swin UNETR consistently outperformed other models, particularly when trained on the 48-band dataset with manually digitised levees. Unlike wide automatic annotations, which produced irregular and noisy patches, thin manual annotations yielded clearer, more linear predictions. Post-processing further refined the outputs, enabling the model to achieve pixel-level precision of 0.6555 and recall of 0.5107, and vector-level precision of 0.7554  and recall of 0.7743  after additional post-processing. While pixel-level metric scores remain modest, reflecting the irregularity of the archaeological features, the model successfully predicted levee networks across ~31,250 km², extending from the Ba’qubah region to Ad Rumaythah, with potentially correctly detected ca. 10,300 km of levees located outside the dataset. Comparative analysis with independent palaeochannel reconstructions confirmed that the model identified many of the most prominent irrigation features while avoiding misclassification of modern infrastructure. The results of the Monte Carlo simulation indicate a clear relationship between identified levees and archaeological sites of different periods, particularly Old Babylonian, Parthian, Sasanian, and Early Islamic.
The results highlight both the challenges and promise of deep learning in archaeological remote sensing. Automated predictions cannot yet replace interpretative digitisation, but they provide reproducible, standardised, and scalable outputs that can accelerate archaeological mapping and support regional-scale analysis. By leveraging multitemporal, multisource datasets and advanced AI architectures, this study demonstrates a pathway towards reconstructing irrigation systems of different historical periods and landscapes. The approach opens new possibilities for documenting, preserving, and interpreting water-management legacies in some of the world’s most significant ancient landscapes. 


## Levee detection demo 

[Open demo](https://github.com/nazarb/Levees_DL/blob/main/Levee_detection_Swin_UNETR_demo_2026.ipynb)

Demo of the model includes:

1. Initiate libraries
2. Download raw data (utilize [Copernicus GLO30](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) and the trained [model](https://danebadawcze.uw.edu.pl/file.xhtml?fileId=17758&version=1.0)
3. Predict the the levees using Swin UNETR model
4. Postprocessing
   
## Full project workflow

![Model](https://github.com/nazarb/Levees_DL/blob/0b6cc1e78e79003e468d97d04d010ed22d7cb4db/images/Diagram.png)

### Setup

The Docker container and virtual envinronment used to run all the scripts is available [here](https://github.com/nazarb/Levees_DL/blob/main/Setup/Setup.md).
More information how to install [Docker](https://docs.docker.com/engine/install/) is provided on the Docker website.

### Pre-processing
The project constisted of two steps. First step was to develop a Deep Learning dataset, which was developed gradually. The final version of the dataset consisted of 20GB. The final steps was the development of the deep learning model.

1. Calculate the multisource rasters using published Google Earth Engine [code](https://github.com/nazarb/Levees_DL/blob/main/Dataset/dataset.md)
2. Create annotations
	1. Rasterize the [levee network](https://doi.org/10.58132/MGOHM8) created for the purpose of this work - [code](https://github.com/nazarb/Levees_DL/blob/b1e94674462cdf34197fe3e1c8e231359777e31f/Pre_processing/1.%20Labels%20-%20convert%20lines%20to%20raster.ipynb)
	2. Clip (QGIS 3.34.13)
	3. Adjust pixels in multisource raster and annotations (QGIS 3.34.13)
3. Devide the images and annotations into tiles of 96x96 pixels - [code](https://github.com/nazarb/Levees_DL/blob/0d97b12ec862e5f634016d1b670492f1acc973c1/Pre_processing/2.%20Split%20into%20tiles.ipynb)
4. Create JSON file with structure of the dataset
	1. Create JSON for one or all rasters - [code](https://github.com/nazarb/Levees_DL/blob/0d97b12ec862e5f634016d1b670492f1acc973c1/Pre_processing/3.%20Create%20a%20structure%20of%20the%20dataset%20in%20JSON.ipynb)
	2. Merge JSON files if necessary - [code](https://github.com/nazarb/Levees_DL/blob/a258868dad8dd6a6d31f031c691e2708c3124224/Pre_processing/4.%20Albumentations/4B1.%20JSON%20merge.ipynb)
6. Perform augmentations using [albumentations](https://github.com/albumentations-team/albumentations) - [code](https://github.com/nazarb/Levees_DL/blob/0d97b12ec862e5f634016d1b670492f1acc973c1/Pre_processing/4.%20Albumentations/augmentation.md)
7. Create an augmented dataset
   1. Append augmented [tiles](https://github.com/nazarb/Levees_DL/blob/a258868dad8dd6a6d31f031c691e2708c3124224/Pre_processing/4.%20Albumentations/4B2.%20JSON%20append%20augmented.ipynb) to a JSON file 
   2. Shuffle the tiles within collections used to train, validate and test the model (with a data leakage prevention mechanism) - [code](https://github.com/nazarb/Levees_DL/blob/a258868dad8dd6a6d31f031c691e2708c3124224/Pre_processing/4.%20Albumentations/4C.%20JSON%20shuffle.ipynb)
   3. Check if the dataset does not have data leakage between train, validation and testing - [code](https://github.com/nazarb/Levees_DL/blob/main/Pre_processing/4.%20Albumentations/4D.%20JSON_checker.ipynb)
  
The dataset structure is provided with following [link](https://github.com/nazarb/Levees_DL/blob/main/Dataset/dataset.md)

### Train and validate the model 
The research utilizes the Unet, Attention Unet and Swin UNETR developed by [MONAI](https://github.com/Project-MONAI/MONAI)
#### Model selection part
The selection phase of the training included separate sets od code for training Unet, Attention Unet and Swin UNETR [models](https://github.com/nazarb/Levees_DL/blob/main/Model_selection/Model_list.md)
- Unet
- Attention Unet
- Swin UNETR
#### The final model

The final version of the model utilized [Swin UNETR](https://github.com/nazarb/Levees_DL/blob/main/Swin_UNETR/Swin_UNETR.md)

The code used in this stage is available using following [link](https://github.com/nazarb/Levees_DL/blob/main/Swin_UNETR/Train.ipynb)

### Predict
The [code](https://github.com/nazarb/Levees_DL/tree/main/Predict) used for prediction consists of several steps
1. Calculate the multisource rasters using published Google Earth Engine [code](https://github.com/nazarb/Levees_DL/blob/main/Dataset/Dataset_creation_GEE_code)
2. Run the detection code
	1. Select raster
	2. Convert to NPY
	3. Predict levees
	4. Convert predictions to TIF and merge them
    
### Post-processing
3. Post-processing [code](https://github.com/nazarb/Levees_DL/blob/main/Post_processing/Post-processing.ipynb) include:
	1. Filter by size (the filter excludes small features) - opencv-python ```connectedComponentsWithStats(); min_area = 350; connectivity=8```
	2. Closing - opencv-python ```Closing; kernel = 7x7```


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




Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
