import os
import cv2
import pandas
from matplotlib import pyplot as plt
import glob


folder = "../ivp-2026/train/train"
images=pandas.read_csv("../ivp-2026/train.csv")

fig, axes = plt.subplots(3, 6, figsize=(15, 12)) 

for i in range(25,35):
    filename = images["ID"][i]
    true_time = images["time"][i]

    path = os.path.join(folder, filename)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax = axes.flat[i-25]         
    ax.imshow(img)
    ax.set_title(true_time)
    ax.axis("off")

plt.tight_layout()             
plt.show()

