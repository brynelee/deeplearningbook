import sys, os
sys.path.append(os.pardir)
sys.path.append(os.curdir)
import numpy as np
from common.functions import mean_squared_error

t = [0, 0, 1, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1]

error = mean_squared_error(np.array(y), np.array(t))
print(error)
