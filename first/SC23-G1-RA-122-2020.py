import sys
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt  


def funkcija(picture_name, correct_result):
    img_original = cv2.cvtColor(cv2.imread(f'{folder_name}{picture_name}'), cv2.COLOR_BGR2RGB)
    
    img = cv2.convertScaleAbs(img_original, alpha=0.95)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_gray = cv2.GaussianBlur(img_gray,(5,5),cv2.BORDER_DEFAULT)

    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37, 10)
    img_bin = img_bin - 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(6,6))
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    img_bin = cv2.erode(img_bin, kernel, iterations=1)
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    img_bin = cv2.erode(img_bin, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    img_bin = cv2.erode(img_bin, kernel, iterations=1)
    img_bin = cv2.dilate(img_bin, kernel, iterations=3)


    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours_real = []

    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        height, width = size
        
        if width > 25 and width < 75 and height > 25 and height < 75 and height/width >= 0.70 and height*width > 42*35:
            contours_real.append(contour)
        
    # image = img.copy()
    # cv2.drawContours(image, contours_real, -1, (255, 0, 0), 1)
    # plt.imshow(image)    
    # plt.show()  
    print(f'{picture_name}-{correct_result}-{len(contours_real)}')
    return abs(correct_result - len(contours_real))


if __name__ == "__main__":
    folder_name = sys.argv[1]
    df = pd.read_csv('squirtle_count.csv')
    results = df['Broj squirtle-ova']
    pictures = df['Naziv slike']
    sum = 0
    for i in range(10):
        sum += funkcija(pictures[i], results[i])

    mae = sum/10
    print(f'{mae}')





