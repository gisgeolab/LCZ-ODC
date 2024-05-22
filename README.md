# Temperature Analysis within Local Climate Zones

This repository provides tools to access temperature measurements from ARPA stations within the metropolitan area of Milan. The goal is to analyse the correlation between temperature readings across different stations to identify potential Urban Heat Island (UHI) effects in Milan.


## Data Access and Interface:

The temperature data are accessed via a local interface called [ROSE-API](http://github.com/Diuke/rose-api), which is not accessible through public networks. Other data include:
-	ARPA Stations Data: Located in the **LCZ.csv** file, this dataset includes the stations' locations and their assigned local climate zones.
-	Metropolitan Area Shapefile: Found in the **CMM.zip**, this shapefile represents the boundaries of the metropolitan area of Milan.
-	Visual Crossing Data: Two auxiliary data frames (for 2022 and 2023) from the Visual Crossing database provide information on sunset times and general weather conditions (e.g., clear, rainy days).


## Data Processing:

A dedicated function for outlier detection and data aggregation is provided in the **ARPA_functions.py** file.


## Analysis Workflow:

- Data Visualization: The notebook starts by showing the distribution and statistics of the temperature data. Users can choose between all measurements or those within specific intervals for more accurate analysis.
-	Correlation Analysis: The analysis involves finding correlations between temperature measurements at different stations with hourly aggregated temperature measurements. The analysis is conducted separately for daytime and nighttime temperatures.

### Statistical Testing:

-	Parametric Test (ANOVA): Used if the data meet normality (assessed via QQ-plots and Shapiro-Wilk test) and homoscedasticity (assessed via Bartlett's test) conditions.
-	Non-Parametric Test (Kruskal-Wallis): Used if the data do not meet the parametric test conditions.

The choice between ANOVA and Kruskal-Wallis is determined by the nature of the data. Users can decide on the normality of the data, with the recommendation to ensure normal distribution within each sample (station measurements).

### Post-Hoc Testing: If significant differences are found:

-	ANOVA: Followed by Tukey's test.
-	Kruskal-Wallis: Followed by Dunn's test.

**Note**: All statistical tests operate under the null hypothesis that there are no differences between the groups' temperature measurements. The p-value threshold for rejecting the null hypothesis is set at 0.05.


---
### Contacts and Authors

Politecnico di Milano DICA Team:
- <b>*Afshin Moazzam*</b> (afshin.moazzam@polimi.it)
- <b>*Alberto Vavassori*</b> (alberto.vavassori@polimi.it)
- <b>*Maria Antonia Brovelli*</b> (maria.brovelli@polimi.it)
- <b>*Daniele Oxoli*</b> (daniele.oxoli@polimi.it)
- <b>*Giovanna Venuti*</b> (giovanna.venuti@polimi.it)

Italian Space Agency (ASI) Team:
- <b>*Deodato Tapete*</b> (deodato.tapete@asi.it)
- <b>*Mario Siciliani de Cumis*</b> (mario.sicilianidecumis@asi.it)
- <b>*Patrizia Sacco*</b> (patrizia.sacco@asi.it)

