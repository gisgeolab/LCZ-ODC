# Pan-sharpening of PRISMA imagery

This Repository contains examples of pan-sharpening methods applied to hyperspectral PRISMA imagery[<sup>1</sup>](#1).

Python codes for pansharpening are provided in the `methods.py` file and accessible through the `Pansharpening.ipynb` Notebook. Functions were implemented in a GitHub Repo[<sup>2</sup>](#2) and are here adapted to the PRISMA imagery context. Metrics for pansharpening quality assessment are provided in the `metrics.py` file, adapted from a dedicated GitHub Repo[<sup>3</sup>](#3).

<ins>Note</ins>: a MATLAB Toolbox is also available for pansharpening of PRISMA data. This toolbox was developed in the context of the *2022 WHISPERS Hyperspectral Pansharpening Challenge*[<sup>4</sup>](#4).

Specifically, the methods implemented in this Repository are based on **Principal Component Analysis**, **Gram-Schmidt Decomposition**, and **Gram-Schmidt Adaptive**, which belong to the Component Substitution methods.

-----

<span id="1">[<sup>1</sup>Loncan, L. et al. **Hyperspectral Pansharpening: A Review.** *IEEE Geoscience and Remote Sensing Magazine* **2015**, 3(3), 1879–1900. doi: 10.1109/MGRS.2015.2440094](https://ieeexplore.ieee.org/document/7284770)</span>

<span id="2">[<sup>2</sup>GitHub Repo for multispectral imagery pansharpening](https://github.com/codegaj/py_pansharpening)</span>

<span id="3">[<sup>3</sup>GitHub Repo for pansharpening quality assessment](https://github.com/wasaCheney/IQA_pansharpening_python)</span>

<span id="4">[<sup>4</sup>Matlab Toolbox for PRISMA pansharpening](https://openremotesensing.net/knowledgebase/hyperspectral-and-multispectral-data-fusion/)</span>

-----

<ins><b>Authors:</b></ins> <b>*Alberto Vavassori*</b> (alberto.vavassori@polimi.it), <b>*Emanuele Capizzi*</b> (emanuele.capizzi@polimi.it) - 2023 - Politecnico di Milano, Italy
