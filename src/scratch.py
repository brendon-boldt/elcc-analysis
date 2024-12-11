"""File to iteratively develop code."""

from importlib import reload
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns  # type: ignore
import matplotlib
import numpy as np

np.set_printoptions(precision=2)
pd.options.display.width = 100
pd.options.display.max_columns = 20

# Prevent warnings about GUI failing in non-main thread for scratchpad; disable
# to use interactive GUI.
matplotlib.use("agg")

from . import core

reload(core)
from .core import *

plt.close()
plt.clf()


# Enter code here that you want to run.


print("Done.")
