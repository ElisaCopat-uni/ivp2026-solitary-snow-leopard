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
    #grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #resize
    h, w = gray.shape
    scale = 500 / w
    resized = cv2.resize(gray, (500, int(h * scale)))
    #canny step
    blurred=cv2.GaussianBlur(resized, (5, 5), 0)

    #canny+hough voting
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1,            # accumulator size = image size
        minDist=500,     # minimum distance between two circle centres - one clock per image
        param1=100,      # Canny's high threshold (strong edges)
        param2=30,       # votes needed: LOWER = more circles found (also fake ones)
        minRadius=50,   # smallest clock radius in pixels 
        maxRadius=250    # biggest clock radius in pixels - one clock per image
    )

    show = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    if circles is not None:
        x, y, r = circles[0][0]           # the best circle is the first one
        x, y, r = int(x), int(y), int(r)  
        cv2.circle(show, (x, y), r, (255, 0, 0), 3)   # circle
        cv2.circle(show, (x, y), 3, (255, 0, 0), -1)  # centre
    else:
        print(filename, "→ no circle found")

    ax = axes.flat[i-25]         
    ax.imshow(show)
    ax.set_title(true_time)
    ax.axis("off")

plt.tight_layout()             
plt.show()