# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:20:07 2020

@author: vaibhav_bhanawat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv');

#implementation of upper confidence bound

d = 10
number_of_selection = [0] * d
sum_of_reward = [0] * d;
ads_selected = []
total_reward = 0
for n in range (0, 10000):
    max_ads  = 0;
    max_upper_bound = 0;
    for j in range (0, 10):
        if number_of_selection[j] > 0:
            average_reward = sum_of_reward[j] / number_of_selection[j]
            upper_bound = average_reward + math.sqrt(3/2 * math.log(n)/number_of_selection [j])
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            max_ads = j
    ads_selected.append(max_ads)
    number_of_selection[max_ads] = number_of_selection[max_ads] + 1
    reward = dataset.values[n, max_ads]
    sum_of_reward[max_ads] = sum_of_reward[max_ads] + reward
    total_reward = total_reward + reward

plt.hist(ads_selected)
plt.title("histogram of Ads selection")
plt.xlabel("Ads")
plt.ylabel("Number of times each Ad was selection")
plt.show()