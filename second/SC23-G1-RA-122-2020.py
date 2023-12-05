import numpy as np
import os
import sys
import cv2
from sklearn.svm import SVC # SVM klasifikator
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def classify_window(window):
    features = hog.compute(window).reshape(1, -1)
    return clf_svm.predict_proba(features)[0][1]


def non_max_suppression_slow(boxes, overlapThresh):
	
	if len(boxes) == 0:
		return []
	
	pick = []
	
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	
	while len(idxs) > 0:
		
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		
		for pos in range(0, last):
			
			j = idxs[pos]
			
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			
			overlap = float(w * h) / area[j]
			
			if overlap > overlapThresh:
				suppress.append(pos)
		
		idxs = np.delete(idxs, suppress)
	
	return boxes[pick]

def find_cars(image, step_size, window_size=(60, 120)):
    best_windows = []
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            this_window = (y, x) # zbog formata rezultata
            window = image[y:y+window_size[1], x:x+window_size[0]]
            if window.shape == (window_size[1], window_size[0]):
                score = classify_window(window)
                if score > 0.975:
                    this_windowFound = (this_window[1], this_window[0], this_window[1] + 60, this_window[0] + 120)
                    best_windows.append(this_windowFound)
    return best_windows

def detect_cross(x, y, k, n):
    yy = k*x + n
    return -85 <= (yy - y) <= 75

def isSameAsLastOne(detectedXs, center_x):
    for dx in detectedXs: 
        if abs(dx - center_x) < 15 :
            return True
    return False

def process_video(video_path, hog_descriptor, classifier):
    sum_of_nums = 0
    k = 0
    n = 0
    
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)

    xMin = 3600
    yMin = 3600
    yMax = -3600
    xMax = -3600
    detectedXsPrev = []
    detectedXsCurr = []
    
    while True:
        
        frame_num += 1
        grabbed, frameNext = cap.read()

        if not grabbed:
            break
        
        
        if frame_num == 1:
            slika = frameNext.copy()
            slika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
            thresh, slika_bin = cv2.threshold(slika, 150, 255, cv2.THRESH_BINARY)
            
            slika_bin = cv2.dilate(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=5)
            slika_bin = cv2.erode(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=8)
            slika_bin = cv2.dilate(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=5)
            slika_bin = cv2.erode(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=5)
            slika_bin = cv2.erode(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=7)
            slika_bin = cv2.dilate(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
            slika_bin = cv2.erode(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=4)
            slika_bin = cv2.dilate(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=5)
            slika_bin = cv2.erode(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=5)
            slika_bin = cv2.erode(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=15)
            slika_bin = cv2.dilate(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)), iterations=8)
            slika_bin = cv2.erode(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=8)
            slika_bin = cv2.dilate(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)), iterations=5)
            slika_bin = cv2.erode(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=15)
            slika_bin = cv2.dilate(slika_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=4)
            contours, _ = cv2.findContours(slika_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_real = []
            for contour in contours:
                center, size, angle = cv2.minAreaRect(contour)
                height, width = size

                if height > 600 and width < 300 and width > 100:
                    contours_real.append(contour)

                if width > 450 and height < 300 and height > 100:
                    contours_real.append(contour)

            
            for c in contours_real:
                center, size, angle = cv2.minAreaRect(c)
                x,y = center
                if x < xMin:
                    xMin = x
                if x > xMax:
                    xMax = x
                if y < yMin:
                    yMin = y
                if y > yMax:
                    yMax = y
            
            frame = frameNext[int(yMin):int(yMax), int(xMin):int(xMax)]
            frame = cv2.resize(frame, (1000, 800), interpolation=cv2.INTER_NEAREST)
            
            edges_img = cv2.Canny(frame, 100, 200, apertureSize=3)
                        
            lines = cv2.HoughLinesP(edges_img, 1, np.pi/180, threshold=100, minLineLength=185, maxLineGap=10)
            
            x1,y1,x2,y2=lines[0][0]

            k = (y2 - y1)/(x2-x1)
            n = y1 - k*x1
            
            continue
        if frame_num == 12 or frame_num == 24:
             continue
        if (frame_num % 12 != 0) and (frame_num != 2):
            if frame_num != 15 and frame_num != 27:
                continue
        
        
        frame = frameNext[int(yMin):int(yMax), int(xMin):int(xMax)]
        frame = cv2.resize(frame, (1000, 800), interpolation=cv2.INTER_NEAREST)
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        b = find_cars(image=frame_gray, step_size=11)

        boundingBoxes = np.array([w for w in b])
        
        pick = non_max_suppression_slow(boundingBoxes, 0.3)

        detectedXsPrev = detectedXsCurr
        detectedXsCurr = []
        for (startX, startY, endX, endY) in pick:

            center_x = startX + (endX-startX) / 2
            center_y = startY - (startY-endY) / 2

            if (detect_cross(center_x, center_y, k, n)):
                if isSameAsLastOne(detectedXsPrev, center_x):
                     continue
                sum_of_nums += 1
                detectedXsCurr.append(center_x)
    cap.release()
    return sum_of_nums


if __name__ == "__main__":
    folder_name = sys.argv[1]
    df = pd.read_csv(f'{folder_name}counts.csv')
    results = df['Broj_prelaza']
    videos = df['Naziv_videa']
    sum = 0

    train_dir = f'{folder_name}/pictures/'

    pos_imgs = []
    neg_imgs = []

    for img_name in os.listdir(train_dir):
        img_path = os.path.join(train_dir, img_name)
        img = load_image(img_path)
        if 'p_' in img_name:
            pos_imgs.append(img)
        elif 'n_' in img_name:
            neg_imgs.append(img)

    pos_features = []
    neg_features = []
    labels = []

    nbins = 9
    cell_size = (8, 8)
    block_size = (3, 3)

    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                    img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    for img in pos_imgs:
        pos_features.append(hog.compute(img))
        labels.append(1)

    for img in neg_imgs:
        neg_features.append(hog.compute(img))
        labels.append(0)

    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)
    x_train = np.vstack((pos_features, neg_features))
    y_train = np.array(labels)

    clf_svm = SVC(kernel='linear', probability=True) 
    clf_svm.fit(x_train, y_train)

    for i in range(len(videos)):
        currentRes = process_video(f'{folder_name}videos/{videos[i]}', hog, clf_svm)
        print(f'{videos[i]}-{results[i]}-{currentRes}')
        sum += abs(results[i] - currentRes)

    mae = sum/len(videos)
    print(f'{mae}')