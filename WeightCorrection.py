from SpectralAnalysis import power_iteration
import numpy as np


W= np.random.randn(5,5)
d,v=power_iteration(W)
print(d)