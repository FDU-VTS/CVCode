# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# a = np.load("./data/preprocessed/test_density/1_3.npy")
a = np.load("./data/result/result/9.npy")
print(a)
plt.imshow(a, cmap="gray")
plt.show()