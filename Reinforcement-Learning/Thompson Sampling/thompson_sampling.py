# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 19:39:23 2020

@author: vaibhav_bhanawat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

#implmemtating Thompson Sampling
ads_selected = []
d = 10
N = 10000
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_rewards = 0
for n in range (0, N):
    selected_ads = 0
    max_random = 0
    for j in range (0, 10):
        random_beta = random.betavariate(numbers_of_rewards_1[j] + 1, numbers_of_rewards_0[j] + 1)
        if random_beta > max_random:
            max_random = random_beta
            selected_ads = j
    ads_selected.append(selected_ads)
    reward = dataset.values[n, selected_ads]
    if reward == 1:
        numbers_of_rewards_1[selected_ads] = numbers_of_rewards_1[selected_ads] + 1
    else:
        numbers_of_rewards_0[selected_ads] = numbers_of_rewards_0[selected_ads] + 1
    total_rewards = total_rewards + reward

plt.hist(ads_selected)
plt.title("Histogram of Ads selection")
plt.xlabel("Ads")
plt.ylabel("Number of times Each Ad selected")
plt.show()