# Pan-sharpening of PRISMA and Sentinel-2 imagery
This Repository contains examples of pan-sharpening methods applied to hyperspectral [PRISMA](https://www.asi.it/scienze-della-terra/prisma/) and multispectral [Sentinel-2](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2) imagery[<sup>1</sup>](#1).

Pan-sharpening is applied to the **Visible and Near-InfraRed** (VNIR) **bands of PRISMA** (66 bands, 30m resolution) and to the **Sentinel-2 bands** (9 band, 20m resolution), leveraging the **panchromatic PRISMA band** (single band, 5m resolution). Images should be co-registered and pre-processed according to the [workflow of the LCZ-ODC project](https://github.com/gisgeolab/LCZ-ODC/tree/Processing-Notebooks).

The functions implementing the pan-sharpening algorithms are provided in [`methods.py`](https://github.com/gisgeolab/LCZ-ODC/blob/Pansharpening/methods.py). This file was produced by adapting the functions available in a dedicated GitHub repository[<sup>2</sup>](#2) to PRISMA and Sentinel-2 images. Specifically, the implemented methods include **Principal Component Analysis** (PCA), **Gram-Schmidt** (GS), and **Gram-Schmidt Adaptive** (GSA), which belong to the category of *component substitution-based algorithms*. Metrics for pan-sharpening quality assessment - implemented by adapting the functions available in a dedicated GitHub repository[<sup>3</sup>](#3) - are provided in [`metrics.py`](https://github.com/gisgeolab/LCZ-ODC/blob/Pansharpening/metrics.py). The [`functions.py`](https://github.com/gisgeolab/LCZ-ODC/blob/Pansharpening/functions.py) file contains ancillary functionalities for data preparation and processing.

Notebooks are structured as follows:
* [`1a - PRISMA-pansharpening.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Pansharpening/1a%20-%20PRISMA-pansharpening.ipynb): pan-sharpening of PRISMA images;
* [`1b - PRISMA-S2-pansharpening.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Pansharpening/1b%20-%20PRISMA-S2-pansharpening.ipynb): pan-sharpening of Sentinel-2 images;
* [`2a - PRISMA-pan_quality.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Pansharpening/2a%20-%20PRISMA-pan_quality.ipynb): assessment of PRISMA pan-sharpened image quality;
* [`2b - PRISMA-S2-pan_quality.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Pansharpening/2b%20-%20PRISMA-S2-pan_quality.ipynb): assessment of Sentinel-2 pan-sharpened image quality.

The complete description of the functionalities is provided within each notebook.

<b>Note</b>: a MATLAB Toolbox is also available for PRISMA image pan-sharpening. This toolbox was developed in the context of the *2022 WHISPERS Hyperspectral Pansharpening Challenge*[<sup>4</sup>](#4).

-----

### Resources

<span id="1">[<sup>1</sup>Loncan, L. et al. **Hyperspectral Pansharpening: A Review.** *IEEE Geoscience and Remote Sensing Magazine* **2015**, 3(3), 1879–1900. doi: 10.1109/MGRS.2015.2440094](https://ieeexplore.ieee.org/document/7284770)</span>

<span id="2">[<sup>2</sup>GitHub Repository for multispectral imagery pansharpening](https://github.com/codegaj/py_pansharpening)</span>

<span id="3">[<sup>3</sup>GitHub Repository for pansharpening quality assessment](https://github.com/wasaCheney/IQA_pansharpening_python)</span>

<span id="4">[<sup>4</sup>Matlab Toolbox for PRISMA pansharpening](https://openremotesensing.net/knowledgebase/hyperspectral-and-multispectral-data-fusion/)</span>

-----

<ins><b>Authors:</b></ins> <b>*Alberto Vavassori*</b> (alberto.vavassori@polimi.it), <b>*Emanuele Capizzi*</b> (emanuele.capizzi@polimi.it), <b>*Vasil Yordanov*</b> (vasil.yordanov@polimi.it) - 2024 - GIS GEOlab - Politecnico di Milano, Italy
