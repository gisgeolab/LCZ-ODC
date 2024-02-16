# Processing, analysis, and classification of PRISMA and Sentinel-2 satellite imagery

This branch contains several Notebooks dedicated to the processing, analysis, and classification of two types of satellite imagery, namely:
* multispectral [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) images of the European Space Agency (ESA);
* hyperspectral [PRISMA](https://www.asi.it/scienze-della-terra/prisma/) (Hyperspectral Precursor of the Application Mission) images of the Italian Space Agency (ASI).

The ultimate goal is to produce maps of Local Climate Zones (LCZs) for the Metropolitan City of Milan.

## Notebooks description

The first two Notebooks are dedicated to data pre-processing. They allow the user to prepare Sentinel-2 and PRISMA data for next stages of processing and analysis. The remaining Notebooks are meant to be used for the exploration, analysis, and classification of ready-to-use data.

### Preprocessing Notebooks
* [`1 - S2_Preprocessing.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/1%20-%20S2_Preprocessing.ipynb): allows the user to read and mosaic the Sentinel-2 tiles covering the area of interest (i.e. the Metropolitan City of Milan). The mosaicked Sentinel-2 image is clipped to the extent of a reference PRISMA image. The functions of this notebook can be adapted also to other tiles. The functions read Sentinel-2 bands in .jpeg2000 format and produce final outputs in GeoTIFF format. The user can select the bands of interest to be exported as well as export an image containing all the available bands.
* [`2 - PRISMA_S2_coregistration.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/2%20-%20PRISMA_S2_coregistration.ipynb): exploits the pre-processed Sentinel-2 images as a reference for PRISMA imagery coregistration using the [GeFolki algorithm](https://github.com/aplyer/gefolki). It is possible to display and coregister both hyperspectral and pancromatic bands. The user is able to export several band combinations of non-coregisted and coregisted PRISMA images for further analyses. It is also possible to visualize the coregistration quality with multiple plots.

### Exploration & Data Analysis Notebooks
* [`3 - Plotting_analysis.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/3%20-%20Plotting_analysis.ipynb): allows the user to interact with pre-processed PRISMA and Sentinel-2 imagery to facilitate data exploration and visualization. For instance, the user can plot the spectral signatures of the training samples, compute statistics describing band correlation, and plot the histogram of specific bands.
* [`3a - Plotting_spectralseparability.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/3a%20-%20Plotting_spectralseparability.ipynb) and [`3b - Plotting_spectralseparability_pan.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/3b%20-%20Plotting_spectralseparability_pan.ipynb): allow the user to plot the spectral signature of training samples and to perform a spectral separability analysis by computing the Jeffries-Matusita distance.
* [`4 - PCA.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/4%20-%20PCA.ipynb): performs a Principal Component Analysis (PCA) of the hyperspectral PRISMA bands using the [scikit-learn](https://scikit-learn.org/stable/index.html) Python library. This is meant to reduce the dimensionality of the dataset while preserving most of the information in the dataset. Reducing the dimensionality enables a faster and potentially more accurate classification. Interactive plots are provided that showcase the results of the PCA. The user can select the number of Principal Components (PCs) to be exported as a multiband GeoTIFF file, where each PC corresponds to a single band.
* [`5 - Classification.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/5%20-%20Classification.ipynb), [`5a - Classification_S2.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/5a%20-%20Classification_S2.ipynb): performs LCZ classification, integrating the urban canopy parameter layers to the PRISMA PCs (or to the Sentinel-2 bands). The user can import training samples in shapefile o geopackage format, where each polygon corresponds to a LCZ class. The classification can be performed using different classification methods available in the [scikit-learn](https://scikit-learn.org/stable/index.html) and [XGBoost](https://xgboost.readthedocs.io/en/stable/) libraries. Methods implemented in this notebook are Random Forest, AdaBoost, Gradient Boosting (*scikit-learn* library), and XGBoost (*XGBoost* library). The code is structured also to perform hyperparameter tuning for a set of user-defined parameters.
* [`6 - Validation.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/6%20-%20Validation.ipynb), [`6a - Validation_S2.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/6a%20-%20Validation_S2.ipynb), and [`6b - Validation_LCZGen.ipynb`](https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/6b%20-%20Validation_LCZGen.ipynb): allows the user to assess the LCZ map accuracy on specified testing samples, provided in shapefile or geopackage format. The code computes the confusion matrix and statistics including overall accuracy, precision, recall, and f1-score.

## Environment setup

It is possible to set up a virtual Python environment using [Anaconda](https://anaconda.org). 

### Install using the .yml file

Open the terminal, move to the folder containing the `environment.yml` file and type: 
```sh
$ conda env create -f environment.yml
```
This command will automatically build the conda environment containing the libraries for data processing and classification.

---
<ins><b>Authors</b></ins>: <b>*Alberto Vavassori*</b> and <b>*Emanuele Capizzi*</b> - Politecnico di Milano, GEOlab.

## Contacts

Politecnico di Milano DICA Team:
- <b>*Maria Antonia Brovelli*</b> (maria.brovelli@polimi.it)
- <b>*Barbara Betti*</b> (barbara.betti@polimi.it)
- <b>*Giovanna Venuti*</b> (giovanna.venuti@polimi.it)
- <b>*Daniele Oxoli*</b> (daniele.oxoli@polimi.it)
- <b>*Alberto Vavassori*</b> (alberto.vavassori@polimi.it)
- <b>*Jesus Rodrigo Cedeno Jimenez*</b> (jesusrodrigo.cedeno@polimi.it)
<!--- - <b>*Emanuele Capizzi*</b> (emanuele.capizzi@polimi.it) -->

Italian Space Agency (ASI) Team:
- <b>*Deodato Tapete*</b> (deodato.tapete@asi.it)
- <b>*Mario Siciliani de Cumis*</b> (mario.sicilianidecumis@asi.it)
- <b>*Patrizia Sacco*</b> (patrizia.sacco@asi.it)
