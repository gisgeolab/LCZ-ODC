# Processing, analysis, and classification of Sentinel-2 and PRISMA satellite imagery

This branch contains several Notebooks which are being developed in the frame of the LCZ-ODC project (funded by the [Italian Space Agency](https://www.asi.it/)).

Notebooks are dedicated to the processing of two types of satellite imagery, namely:
* multispectral [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) images of the European Space Agency (ESA);
* hyperspectral [PRISMA](https://www.asi.it/scienze-della-terra/prisma/)[<sup>1</sup>](#1) images of the Italian Space Agency (ASI).

The project is mainly focused on the exploitation of PRISMA data to compute Local Climate Zone (LCZ) maps of the study region. Nonetheless, Sentinel-2 data are used as a reference to co-register PRISMA images.

The area of interest is the Metropolitan City of Milan (MCM) (Northern Italy).

## Notebooks description

1. `2S_Preprocessing.ipynb`: allows to read and mosaic the Sentinel-2 tiles covering the area of interest (Metropolitan City of Milan in our case study). The mosaicked Sentinel-2 is clipped the reference PRISMA image extent. The functions of this notebook can be adapted also for other tiles. It is able to read Sentinel-2 bands in .jpeg2000 format and produce final outputs in GeoTIFF format. The user is able to select which bands he wants to export or export the image containing the whole number of bands available.
2. `PRISMA_S2_coregistration.ipynb`: exploits the pre-processed Sentinel-2 images as a reference for PRISMA imagery coregistration using the [GeFolki algorithm](https://github.com/aplyer/gefolki). It is possible to display and coregister both hyperspectral and pancromatic bands. The user is able to export several band combinations of non-coregisted and coregisted PRISMA images for further analyses. Visualizations about the quality of the coregistration are also provided.
3. `Plotting.ipynb`: allows the user to interact with pre-processed PRISMA imagery in order to provide a better description of these data. This is done for example by plotting median spectral signatures for several classes available in the training samples, and computing statistics such as band correlation and bands histograms.
4. `PCA.ipynb`: performs a Principal Component Analysis (PCA) of the hyperspectral PRISMA bands using the [scikit-learn](https://scikit-learn.org/stable/index.html) Python library. This is done in order to reduce the dimensionality of the dataset, but keeping the highest explained variance in the dataset. Reducing the dimensionality allows a faster and better classification. The PRISMA image transformed using PCA can be exported in GeoTIFF file, where each PC corresponds to a single band. The user is able to select the number of bands to be saved in the GeoTIFF file.
5. `Classification.ipynb`: performs a classification of PRISMA Principal Components into Local Climate Zones. The user is able to provide training samples in shapefile o geopackage format (vectorial), where each polygon corresponds to a specific LCZ class. The classification can be performed using different classification methods available in the [scikit-learn](https://scikit-learn.org/stable/index.html) library and the [XGBoost](https://xgboost.readthedocs.io/en/stable/) library. The implemented methods are:

    - Random Forest - scikit-learn
    - AdaBoost - scikit-learn
    - Gradient Boosting - scikit-learn
    - XGBoost

    The code is structured also to perform cross-validation using a set of user defined parameters, that can be eventually modified depending on the user's needs.
    The evaluation of the accuracy is also provided.
6. `Validation.ipynb`: allows to validate the quality of the classification by using an external dataset never used during the classification step. These are called testing samples and must be provided by the user.

## Environment setup

It is possible to set up a virtual Python environment using [Anaconda](https://anaconda.org). 

### Install using the .yml file

Open the terminal, move to the folder containing the `environment.yml` file and type: 
```sh
$ conda env create -f environment.yml
```
This command will automatically build the conda environment containing the libraries for data processing and features selection.

## Notes

<span id="1"><sup>1</sup>Hyperspectral Precursor of the Application Mission (*Precursore Iperspettrale della Missione Applicativa*)</span>

---
## Contacts and Authors

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
