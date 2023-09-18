# Processing, analysis, and classification of PRISMA and Sentinel-2 satellite imagery

This branch contains several Notebooks which are being developed in the frame of the LCZ-ODC project. 

Notebooks are dedicated to the processing of two types of satellite imagery, namely:
* multispectral [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) images of the European Space Agency (ESA);
* hyperspectral [PRISMA](https://www.asi.it/scienze-della-terra/prisma/) (Hyperspectral Precursor of the Application Mission) images of the Italian Space Agency (ASI).

The LCZ-ODC project is mainly focused on the exploitation of PRISMA data to compute Local Climate Zone (LCZ) maps. Nonetheless, Sentinel-2 data are used as a reference to co-register PRISMA images.

## Notebooks description

The first two Notebooks are dedicated to data pre-processing. They allow the user to prepare Sentinel-2 and PRISMA data for next stages of processing and analysis. The remaining Notebooks are meant to be used by end-users for the exploration, analysis, and classification of ready-to-use data.

### Preprocessing Notebooks
1. [`S2_Preprocessing.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/f7cbc26f8995a6d680135c6b0295c554be053633/1%20-%20S2_Preprocessing.ipynb): allows the user to read and mosaic the Sentinel-2 tiles covering the area of interest (i.e. the Metropolitan City of Milan). The mosaicked Sentinel-2 image is clipped to the extent of a reference PRISMA image. The functions of this notebook can be adapted also for other tiles. The functions read Sentinel-2 bands in .jpeg2000 format and produce final outputs in GeoTIFF format. The user can select the bands of interest to be exported as well as export an image containing all the available bands.
2. [`PRISMA_S2_coregistration.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/f7cbc26f8995a6d680135c6b0295c554be053633/2%20-%20PRISMA_S2_coregistration.ipynb): exploits the pre-processed Sentinel-2 images as a reference for PRISMA imagery coregistration using the [GeFolki algorithm](https://github.com/aplyer/gefolki). It is possible to display and coregister both hyperspectral and pancromatic bands. The user is able to export several band combinations of non-coregisted and coregisted PRISMA images for further analyses. Visualizations about the quality of the coregistration are also provided.

### Exploration & Data Analysis Notebooks
3. [`Plotting.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/f7cbc26f8995a6d680135c6b0295c554be053633/3%20-%20Plotting.ipynb
): allows the user to interact with pre-processed PRISMA and Sentinel-2 imagery to facilitate data exploration and visualization. For instance, the user can plot median spectral signatures of multiple classes available in the training samples, compute statistics describing band correlation, and plot the histogram of specific bands.
4. [`PCA.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/f7cbc26f8995a6d680135c6b0295c554be053633/4%20-%20PCA.ipynb
): performs a Principal Component Analysis (PCA) of the hyperspectral PRISMA bands using the [scikit-learn](https://scikit-learn.org/stable/index.html) Python library. This is meant to reduce the dimensionality of the dataset while preserving most of the information in the dataset. Reducing the dimensionality enables a faster and potentially more accurate classification. Interactive plots are provided that showcase the results of the PCA. The user can select the number of Principal Components (PCs) to be exported as a multiband GeoTIFF file, where each PC corresponds to a single band.
5. [`Classification.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/641b6e06f8a25003b64eace821631a9f3fca1494/5%20-%20Classification.ipynb
): performs a classification of PRISMA PCs into LCZs. The user can import training samples in shapefile o geopackage format, where each polygon corresponds to a LCZ class. The classification can be performed using different classification methods available in the [scikit-learn](https://scikit-learn.org/stable/index.html) and [XGBoost](https://xgboost.readthedocs.io/en/stable/) libraries. Methods implemented in this notebook are Random Forest, AdaBoost, Gradient Boosting (*scikit-learn* library), and XGBoost (*XGBoost* library). The code is structured also to perform cross-validation using a set of user-defined parameters, that can be eventually modified depending on the user's needs. Training samples are split into training and testing set in order to provide an assessment of classification accuracy.
6. [`Validation.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/f7cbc26f8995a6d680135c6b0295c554be053633/6%20-%20Validation.ipynb): allows the user to validate the quality of the classification by using an external and independent dataset (i.e. testing samples) that can be provided by the user.

## Environment setup

It is possible to set up a virtual Python environment using [Anaconda](https://anaconda.org). 

### Install using the .yml file

Open the terminal, move to the folder containing the `environment.yml` file and type: 
```sh
$ conda env create -f environment.yml
```
This command will automatically build the conda environment containing the libraries for data processing and classification.

---
<ins><b>Authors</b></ins>: <b>*Alberto Vavassori*</b> and <b>*Emanuele Capizzi*</b> - Politecnico di Milano, 2023.

## Contacts

Politecnico di Milano DICA Team:
- <b>*Maria Antonia Brovelli*</b> (maria.brovelli@polimi.it)
- <b>*Barbara Betti*</b> (barbara.betti@polimi.it)
- <b>*Giovanna Venuti*</b> (giovanna.venuti@polimi.it)
- <b>*Daniele Oxoli*</b> (daniele.oxoli@polimi.it)
- <b>*Alberto Vavassori*</b> (alberto.vavassori@polimi.it)
- <b>*Jesus Rodrigo Cedeno Jimenez*</b> (jesusrodrigo.cedeno@polimi.it)
- <b>*Emanuele Capizzi*</b> (emanuele.capizzi@polimi.it)

Italian Space Agency (ASI) Team:
- <b>*Deodato Tapete*</b> (deodato.tapete@asi.it)
- <b>*Mario Siciliani de Cumis*</b> (mario.sicilianidecumis@asi.it)
- <b>*Patrizia Sacco*</b> (patrizia.sacco@asi.it)
