import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
import warnings
from sklearn.datasets import load_boston
price_dataset = load_boston()

print(price_dataset.keys())
print(price_dataset.DESCR)

boston = pd.DataFrame(price_dataset.data, columns=price_dataset.feature_names)
boston.head()

