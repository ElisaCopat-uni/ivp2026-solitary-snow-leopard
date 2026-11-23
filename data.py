import os
import cv2
import pandas
from matplotlib import pyplot as plt
import glob


def time_to_min(time):
    h,m=time.split(":")
    h=int(h)
    h=h%12
    m=int(m)
    return h*60+m

def time_difference(ta,tb):
    a=time_to_min(ta)
    b=time_to_min(tb)
    diff=abs(a-b)
    diff=min(diff,720-diff)
    return diff

def score(ta,tb):
    x=time_difference(ta,tb)
    if x<=2:
        return 1.0
    elif x<=10:
        return 0.8
    elif x<=30:
        return 0.5
    else:
        return 0.0

if __name__ == "__main__":
    print(score("03:13", "03:13"))
    print(score("03:00", "03:20"))

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
    #plt.show()