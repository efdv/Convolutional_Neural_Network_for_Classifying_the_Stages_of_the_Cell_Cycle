# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:25:58 2023

@author: ef.duquevazquez
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


general_root = '../info/'
model_wgan_root = general_root + 'ch3-wgangp/'


name_model_wgan = 'result_model_0.xlsx'
df = pd.read_excel(model_wgan_root+name_model_wgan, index_col='kfold')
df2 = df.drop(columns= [ "phase", "precision", "recall", "weighted_avg"])

sns.set(style="darkgrid")

df2.plot(kind='bar')

# sns.barplot(data=df2)

plt.show()
