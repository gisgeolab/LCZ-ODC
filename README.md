## Project description
...
WP and outputs description.
Link to:
- Docker ODC
- Processing Notebooks
---
Authors/ASI/POLIMI


# LCZ-ODC Project (Data-driven moDelling of particUlate with Satellite Technology aid)

## Project description
The LCZ-ODC (Local Climate Zone - Open Data Cube) project aims to develop an innovative methodology for LCZ mapping by leveraging multiple data sources and state-of-the-art technologies for geospatial data management. The testbed selected for the activities is the Metropolitan City of Milan (northern Italy). 

<p align="center">
<img src=img/testbed.png width="600">
</p>

The D-DUST project aims at assessing the contribution (in terms of data availability, operability, cost-effectiveness and accuracy improvement) deriving by the systematic integration of different data sources into the current PM monitoring services of the Lombardy region.

This repository contains the code and the information related to the following Work Packages, organized in the following branches:

1. [WP2 - Data Packages](https://github.com/gisgeolab/D-DUST/tree/WP2)
    - State-of-the-art air quality and PM monitoring: the output of this point is a review paper published in an international journal
    - Data repository containing the output files
2. [WP4 - Predictive Model Design](https://github.com/gisgeolab/D-DUST/tree/WP4)
    - State-of-the-art of PM modelling and prediction
    - Models development

## 1. WP2 - Data Packages

> :warning: More information about the notebooks structure and the considered data are provided in the **[WP2 branch](https://github.com/gisgeolab/D-DUST/tree/WP2)**.

The D-DUST Work Package 2 focuses on the implementation of the project Analysis-ready Data Repository. Data include satellite-based estimates of PM and precursors gases, periodical high-detailed PM observations from on-site sampling with chemical characterization, and ancillary GIS (Geographic Information Systems) data to account for local territorial features that can be informative on PM emissions. Vector grids containing the identified variables are the final output of the the WP2.

Summarizing, the following notebooks have been developed for data preparation and processing:

- [**Model Variables Request Notebook**](https://github.com/opengeolab/D-DUST/blob/WP2/Model%20Variables%20Request.ipynb): this notebook is used to retrieve modelled air quality data.
- [**Ground Sensor Variables Request**](https://github.com/gisgeolab/D-DUST/blob/WP2/Ground%20Sensor%20Variables%20Request%20.ipynb) : this notebook is used to retrieve data for both meteorological and air quality ground stations.
- [**Satellite Variables Request**](https://github.com/opengeolab/D-DUST/blob/WP2/Satellite%20Variables%20Request.ipynb): this notebook is used to retrieve satellite observations of atmospheric pollutants.
- [**Date selection**](https://github.com/opengeolab/D-DUST/blob/WP2/Date%20selection.ipynb): this notebook is used to select low precipitation and high-temperature periods.
- [**Grid Processing**](https://github.com/gisgeolab/D-DUST/blob/WP2/Grid%20Processing.ipynb): this notebook allows computing summary statistics for each variable in each grid cell.
- Features Selection: add to WP2 branch. Dashboard created to select relevant features using multiple statistical methods, in order to be used in the next modeling steps. (fix)

Link to Zenodo data and review paper

## 2. WP4 - Predictive Model Design

> :warning: More information about the notebooks structure and the considered data are provided in the **[WP4 branch](https://github.com/gisgeolab/D-DUST/tree/WP4)**.

The data repository created in the WP2, will be used in the WP4 concerning monitoring and prediction models development. The use of machine-learning models aims at improving the prediction accuracies into local PM analysis by enabling statistical regressors to account for ancillary spatial covariates affecting local PM emissions.

---

<ins><b>Authors</b></ins>: <b>*Daniele Oxoli*</b> (daniele.oxoli@polimi.it), <b>*Emanuele Capizzi*</b> (emanuele.capizzi@polimi.it), <b>*Lorenzo Gianquintireri*</b> (lorenzo.gianquintieri@polimi.it), <b>*Jesus Rodrigo Cedeno Jimenez*</b> (jesusrodrigo.cedeno@polimi.it), <b>*Matteo Bresciani*</b> (matteo.bresciani@mail.polimi.it) - Politecnico di Milano, 2022.


