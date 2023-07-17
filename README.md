# LCZ-ODC Project (Local Climate Zones - Open Data Cube)

## Project description
The LCZ-ODC (Local Climate Zone - Open Data Cube) project aims to develop an innovative methodology for LCZ mapping by leveraging multiple data sources and state-of-the-art technologies for geospatial data management. The testbed selected for the activities is the Metropolitan City of Milan in northern Italy. 

<p align="center">
<img src=img/cmm.png width="600">
</p>

The main purpose of the LCZ-ODC project is the dentification of Local Climate Zones (LCZs) and the study of their correlation with air temperature in the Metropolitan City of Milan through the integration of geospatial data and Earth Observation technologies in an Open Data Cube (ODC) environment. The exclusive use of free and open-source software is a key paradigm of the project for both data processing and analysis.

LCZ are identified by integrating high spectral and spatial resolution satellite imagery obtained from the [PRISMA hyspectral satellite](https://www.asi.it/en/earth-science/prisma/) and [Sentinel-2 multispectral satellite](https://sentinel.esa.int/web/sentinel/missions/sentinel-2). Moreover, additional regional and local geospatial data are exploited for their classification.

The [ODC](https://www.opendatacube.org/) is used in the project as a backend system for managing multi-dimensional and multi-temporal geospatial data with heterogeneous format and resolution in a single end-point. This software provides access to structurally complex files that alternatively would require high expertise from the user. Accordingly, the user can easily access to ready-to-use data for next stages of processing or analysis.

The project also aims to study the correlation between LCZs and the urban heat island. To achieve this, meteorological data provided by regional monitoring networks (e.g. [ARPA Lombardia](https://www.arpalombardia.it/)) available for the Milan Metropolitan City are used and integrated into the ODC.

This project is being developed in collaboration between **Italian Space Agency (ASI)** and **Politecnico di Milano, Department of Civil and Environmental Engineering (POLIMI DICA)**, within the context of the I4DP_SCIENCE program.

### I4DP_SCIENCE Program
The LCZ-ODC project is an integral part of the [**I4DP_SCIENCE**](https://www.asi.it/bandi_e_concorsi/call-for-ideas-i4dp_science-innovation-for-downstream-preparation-for-science/) program of ASI (agreement n. 2022-30-HH.0). The program serves as an incubation platform to demonstrate, in collaboration with ASI, the operational use of scientifically and operationally mature methods and algorithms, equipped with credible performance/capacity. These methods and algorithms aim to address the needs of the user community that are currently partially satisfied or not yet fulfilled.

You can find more information on the following websites:
-  [ASI webpage - LCZ ODC Project First results](https://www.asi.it/2023/05/i4dp_science-primi-traguardi-del-progetto-lcz-odc/)
- [POLIMI DICA webpage - LCZ ODC project](https://www.dica.polimi.it/asi-e-dica-al-via-il-progetto-lcz-odc-una-nuova-frontiera-per-lanalisi-climatica-urbana/)

### LCZ-ODC Repository
This Repository contains the code and information related to the Work Package 2 (WP2) *Development* of the LCZ-ODC project. It includes the ODC Docker container and the LCZ Processing Notebooks.
The **branches** in this repository allows you to access the code and the information related to WP2, and they are organized as follows:

1. [Docker-ODC](https://github.com/gisgeolab/LCZ-ODC/tree/Docker-ODC)
    - This branch contains the ODC Docker container.
    - You can find documentation related to the ODC Docker setup and usage.
2. [Processing-Notebooks](https://github.com/gisgeolab/LCZ-ODC/tree/Processing-Notebooks)
    - This branch contains the Notebooks used for data preprocessing and obtaining analysis-ready data.
    - You can find Notebooks for performing LCZ classification and validation.
    - Additionally, there are notebooks available for data exploration, visualization, and analysis.

<br>

> :warning: NOTE: In addition, a QGIS plugin called ARPA Weather has been developed in the context of LCZ-ODC project, in order to retrieve ARPA Lombardia ground sensors. More informations are provided in the dedicated [ARPA Weather Plugin - Github repository](https://github.com/gisgeolab/ARPA_Weather_plugin).

A schematic overview of the software architecture of the LCZ-ODC project is depicted in the following figure.

<p align="center">
<img src=img/architecture.png width="350">
</p>

---
### Contacts and Authors

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

