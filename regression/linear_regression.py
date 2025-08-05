import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
import warnings
from sklearn.datasets import load_boston
price_dataset = load_boston()

print(price_dataset.keys())
print(price_dataset.DESCR)



lasso_alpha_list=np.linspace(1e-5,0.1,500)
lasso_alpha_acc_results=[]
for i in lasso_alpha_list:
    lasso_regression(i, lasso_alpha_acc_results)
lasso_results=dict(zip(lasso_alpha_list,lasso_alpha_acc_results))
print("Lasso optimal alpha:",max(lasso_results,key=lasso_results.get),", accuracy:",max(lasso_results.values()))


