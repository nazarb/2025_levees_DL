# Swin UNETR levee detection model

Buławka, Nazarij, Hector A. Orengo, Felipe Lumbreras Ruiz, Iban Berganzo-Besga and Ekta Gupta. n.d. ‘Leveraging Big Multitemporal Multisource Satellite Data and Artificial Intelligence for the Detection of Complex and Invisible Features - the Case of Extensive Irrigation Mapping’.

*In construction*

## Abstract

The detection of buried or obscured archaeological features remains a central challenge in landscape archaeology, particularly in the irrigated floodplains of Mesopotamia where levees and canals formed the basis of complex agrarian systems. This study presents a deep learning–based approach for the large-scale, semi-automated detection of ancient levees in central Iraq, integrating big multitemporal and multisource satellite datasets with advanced instance segmentation models.
Datasets were assembled from multitemporal Landsat 5, Sentinel-1 SAR, Sentinel-2 multispectral imagery, and the TanDEM-X Edited DSM, combined with vegetation and moisture indices, PCA reductions of seasonal variability, and Multi-Scale Relief Model (MSRM) outputs. Training labels were generated through both threshold-based automatic extraction and detailed manual digitisation. Three architectures: U-Net, Attention U-Net, and Swin UNETR, were evaluated on datasets containing 53, 48, and 36 bands.
Results demonstrate that Swin UNETR consistently outperformed other models, particularly when trained on the 48-band dataset with manually digitised levees. Unlike wide automatic annotations, which produced irregular noisy patches, thin manual annotations yielded clearer, more linear predictions. Post-processing further refined outputs, allowing the model to achieve precision of 0.5118 and recall of 0.5908 at the pixel level. While metric scores remain modest, reflecting the irregularity of the archaeological features, the model successfully predicted levee networks across ~23,600 km², extending from Babylon to Uruk. Comparative analysis with independent palaeochannel reconstructions confirmed that the model identified many of the most prominent irrigation features while avoiding misclassification of modern infrastructure.
The results highlight both the challenges and promise of deep learning in archaeological remote sensing. Automated predictions cannot yet replace interpretative digitisation, but they provide reproducible, standardised, and scalable outputs that can accelerate archaeological mapping and support regional-scale analysis. By leveraging multitemporal, multisource datasets and advanced AI architectures, this study demonstrates a pathway towards reconstructing irrigation systems of different historical periods and landscapes. The approach opens new possibilities for documenting, preserving, and interpreting water-management legacies in some of the world’s most significant ancient landscapes. 


## Workflow


## Citation

```bash


@article{bulawkaLeveragingBigMultitemporal01,
	title = {Leveraging big multitemporal multisource satellite data and artificial intelligence for the detection of complex and invisible features - the case of extensive irrigation mapping},
	volume = {},
	doi = {},
	number = {},
	journal = {GIScience & Remote Sensing},
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
