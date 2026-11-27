import os
import cv2
import pandas
from matplotlib import pyplot as plt
import glob
import numpy as np
from data import score



mode = "train"    # "train" = score on train, "test" = make the Kaggle file
folder = f"../ivp-2026/{mode}/{mode}"
images = pandas.read_csv(f"../ivp-2026/{mode}.csv")
results = []

fig, axes = plt.subplots(3, 6, figsize=(15, 12)) 

def group_hands(hand_lines, max_diff=10):
    groups = []
    for angle, length in sorted(hand_lines):
        if groups and angle - groups[-1][-1][0] < max_diff:
            groups[-1].append((angle, length))
        else:
            groups.append([(angle, length)])
    hands = []
    for g in groups:
        hands.append(max(g, key=lambda line: line[1]))
    hands.sort(key=lambda hand: hand[1], reverse=True)
    return hands

total = 0
for i in range(len(images)):
    filename = images["ID"][i]
    if mode == "train":
        true_time = images["time"][i]
    else:
        true_time = "?"

    path = os.path.join(folder, filename)
    img = cv2.imread(path)
    #grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #resize
    h, w = gray.shape
    scale = 500 / w
    resized = cv2.resize(gray, (500, int(h * scale)))
    #canny step
    blurred=cv2.GaussianBlur(resized, (9, 9), 0)
    edges = cv2.Canny(blurred, 50, 150)

    #canny+hough voting
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1,            # accumulator size = image size
        minDist=500,     # minimum distance between two circle centres - one clock per image
        param1=200,      # Canny's high threshold (strong edges) -> 100 gives 7/10
        param2=40,       # votes needed: LOWER = more circles found (also fake ones)
        minRadius=30,   # smallest clock radius in pixels 
        maxRadius=250    # biggest clock radius in pixels - one clock per image
    )
    if(circles is None):
        circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=1,            # accumulator size = image size
                minDist=500,     # minimum distance between two circle centres - one clock per image
                param1=200,      # Canny's high threshold (strong edges) -> 100 gives 7/10
                param2=25,       # votes needed: LOWER = more circles found (also fake ones)
                minRadius=30,   # smallest clock radius in pixels 
                maxRadius=250    # biggest clock radius in pixels - one clock per image
            )

    show = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    if circles is not None:
        x, y, r = circles[0][0]           # the best circle is the first one
        x, y, r = int(x), int(y), int(r)  
        hand_lines = []
        cv2.circle(show, (x, y), r, (255, 0, 0), 3)   # circle
        cv2.circle(show, (x, y), 3, (255, 0, 0), -1)  # centre

        #edges 
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=8, maxLineGap=5)
        #cv2.line(show, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        if lines is not None:
                for x1, y1, x2, y2 in lines.reshape(-1, 4):
                    d1 = np.hypot(x1 - x, y1 - y)   
                    d2 = np.hypot(x2 - x, y2 - y)  
                    if min(d1,d2) <= 0.35*r and max(d1,d2) <= r: 
                        cv2.line(show, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if d1 > d2:
                            tip_x, tip_y = x1, y1
                        else:
                            tip_x, tip_y = x2, y2
                        dx = tip_x - x
                        dy = tip_y - y
                        angle = np.degrees(np.arctan2(dx, -dy)) % 360
                        hand_lines.append((angle, max(d1, d2)))
        print(filename, [(round(a), round(l)) for a, l in hand_lines])
        #decide the estimated time
        hands = group_hands(hand_lines)
        if len(hands) >= 2:
            minute_angle = hands[0][0]  
            hour_angle = hands[1][0]    
        elif len(hands) == 1:
            minute_angle = hands[0][0]   # hands overlap
            hour_angle = hands[0][0]
        else:
            minute_angle = 0             #guess 12:00
            hour_angle = 0
        minute=round(minute_angle/6)%60
        hour = round((hour_angle - minute * 0.5) / 30) % 12
        if hour == 0:
            hour = 12
        #print("  hands:", [(round(a), round(l)) for a, l in hands])
        pred = f"{hour:02d}:{minute:02d}"
        print("  predicted:", pred, "true:", true_time)
        
    else:
        print(filename, "→ no circle found")
        pred = "12:00"

    if mode == "train":
        total += score(pred, true_time) #scoring
    results.append((filename, pred))

    if i < 18:
        ax = axes.flat[i]         
        ax.imshow(show)
        ax.set_title(true_time)
        ax.axis("off")

if mode == "train":
    print("average score:", total / len(images))
else:
    pandas.DataFrame(results, columns=["ID", "time"]).to_csv("../submission.csv", index=False)
    print("saved ../submission.csv with", len(results), "rows")
plt.tight_layout()             
plt.show()