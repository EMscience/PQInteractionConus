[![DOI](https://zenodo.org/badge/425992470.svg)](https://zenodo.org/badge/latestdoi/425992470)
## Data and script for:
"Strength and memory of precipitation's control over streamflow across the conterminous United States"

The CAMELS dataset can be obtained from: https://ral.ucar.edu/solutions/products/camels 
## Folders and descriptions:
	1. CAMELS - contains the information flow metrics based on the CAMELS dataset.
	2. HBV - contains the dataset for the two watersheds (French Broad and Guadalupe).
	3. Code - contains the main script that computes the information flow metrics.

The information flow metrics include:

1. Entropies - $ H(X) $ and $ H(Y) $
2. Mutual information - $I(Y_{t};X_{t-lag})$
3. Transfer entropy - $I(Y_{t};X_{t-lag}|Y_{t-1})$
4. Correlation coeficients - $\r (Y_{t},X_{t-lag})$
5. Total information based on X - $I(Y_{t}; X_{t},X_{t-1})$
6. Total information based on Y - $I(Y_{t}; X_{t},Y_{t-1})$


\
*Edom Moges* \
*edom.moges@berkeley.edu* \
*[Environmental Systems Dynamics Laboratory (ESDL)](https://www.esdlberkeley.com/)*\
*University of California, Berkeley* 