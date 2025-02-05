# TAGM

This is a python implementation of the T-augmented Gaussian Mixture model used to extract protein intracellular localisation and organelle assignments from high-throughput proteomics data (LOPIT-DC, HyperLOPIT, etc.) [1, 2].

The original version of this was implemented in R, as part of the pRoloc package. This is a python port, implementing the scikit-learn API.

## Usage

```python
import numpy as np
from tagm import TAGMMAP

# Create artificial test data with two feature dimensions.
# We use three clusters, one at the origin, one centered at (3, 3) and one centered at (-3, 3) with 100, 50 and 10 data points each.
X = np.vstack([
    np.random.normal(size=(100, 2)),
    np.random.normal(size=(50, 2)) + np.array([3, 3])[np.newaxis],
    np.random.normal(size=(10, 2)) + np.array([-3, 3])[np.newaxis],
])

y = np.zeros((X.shape[0], 3), dtype=bool)
# Five markers for the first cluster...
y[0, :5] = 1
# ... six markers for the second...
y[1, 100:106] = 1
# ... and five markers for the third.
y[2, 150:155] = 1

# Fit the model and predict the localisations
localisations = TAGMMAP().fit_predict(X, y)
```

## References

1. Crook OM, Mulvey CM, Kirk PDW, et al. A Bayesian mixture modelling approach for spatial proteomics. PLoS Computational Biology 14(11):e1006516 2018. https://doi.org/10.1371/journal.pcbi.1006516
2. Crook OM, Breckels LM, Lilley KS et al. A Bioconductor workflow for the Bayesian analysis of spatial proteomics. F1000Research 2019, 8:446. https://doi.org/10.12688/f1000research.18636.1
