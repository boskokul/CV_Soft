import numpy as np
import cv2 # OpenCV
import matplotlib
import matplotlib.pyplot as plt
import collections
import math
from scipy import ndimage
import sys
import pandas as pd
from sklearn.cluster import KMeans

# keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

from tensorflow.keras.optimizers import SGD

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 30, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def dilate(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)), iterations=1)

def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def select_roi(image_orig, img):
    image_bin = erode(dilate(img))
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po X osi
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 310 and h < 80 and w < 80 and h > 39 and w > 23:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
            additionalDims = 0, 0, 0, 0
            for contour in contours:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                if (x1 < x + 8 and x1 > x - 8) and y > y1 :
                    # cv2.rectangle(image_orig, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
                    additionalDims = x1,y1,w1,h1
            region = image_bin[y - additionalDims[3]:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h + additionalDims[3])])
            # cv2.rectangle(image_orig, (x, y - additionalDims[3]), (x + w, y + h), (0, 255, 0), 2)
    
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions

def select_roi_with_distances(image_orig, img):
    image_bin = erode(dilate(img))
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 310 and h < 80 and w < 80 and h > 39 and w > 23:
            additionalDims = 0, 0, 0, 0
            for contour in contours:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                if (x1 < x + 8 and x1 > x - 8) and y > y1 :
                    # cv2.rectangle(image_orig, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
                    additionalDims = x1,y1,w1,h1
            region = image_bin[y - additionalDims[3]:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h + additionalDims[3])])
            # cv2.rectangle(image_orig, (x, y - additionalDims[3]), (x + w, y + h), (0, 255, 0), 2)
    
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # izdvojiti sortirane parametre opisujucih pravougaonika
    # izracunati rastojanja izmedju svih susednih regiona po X osi i dodati ih u niz rastojanja
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2]) # x_next - (x_current + w_current)
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32) # dati ulaz
    y_train = np.array(y_train, np.float32) # zeljeni izlazi na date ulaze
    
    # print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    # print("\nTraining completed...")
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result_as_word(outputs, alphabet):
    result = alphabet[winner(outputs[0])]
    # iterativno dodavanje prepoznatih elemenata
    for idx, output in enumerate(outputs[1:, :]):
        result += alphabet[winner(output)]
    return result

def display_result_with_spaces(outputs, alphabet, k_means):
    # odredjivanje indeksa grupe koja odgovara rastojanju izmedju reci
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    # iterativno dodavanje prepoznatih elemenata
    # dodavanje space karaktera ako je rastojanje izmedju dva slova odgovara rastojanju izmedju reci
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result

#sa stackoverlow-a
def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

if __name__ == "__main__":
    folder_name = sys.argv[1]
    df = pd.read_csv(f'{folder_name}res.csv')
    captchaNames = df['file']
    results = df['text']

    allLettersRegions = []

    for i in range(1, 11, 1):
        image_color = load_image(f'{folder_name}pictures/captcha_{i}.jpg')
        img = image_bin(image_gray(image_color))
        selected_regions, characters = select_roi(image_color.copy(), img)
        allLettersRegions.append(characters)
        # display_image(selected_regions)
        # plt.show()
        
    allLettersRegionsArray = []
    for pictureRegions in allLettersRegions:
        for region in pictureRegions:
            allLettersRegionsArray.append(region)

    letters = ''
    for letter in results:
        letters += letter
    
    letters = letters.replace(' ', '')

    alphabet = []

    lettersDone = set()
    lettersRegionsUnique = []

    # print(len(letters), len(allLettersRegionsArray))
    for l, r in zip(letters, allLettersRegionsArray):
        if l in lettersDone:
            continue
        alphabet.append(l)
        lettersRegionsUnique.append(r)
        lettersDone.add(l)

    inputs = prepare_for_ann(lettersRegionsUnique)
    outputs = convert_output(alphabet)
    ann = create_ann(output_size=len(outputs))
    ann = train_ann(ann, inputs, outputs, epochs=1000)
    # print(alphabet)
    # print(letters)

    errorSum = 0
    for i in range(1, 11, 1):
        image_color = load_image(f'{folder_name}pictures/captcha_{i}.jpg')
        img = image_bin(image_gray(image_color))
        selected_regions, letters, distances = select_roi_with_distances(image_color.copy(), img)
        # print("Broj prepoznatih regiona: ", len(letters))
        # display_image(selected_regions)
        # plt.show()

        inputs = prepare_for_ann(letters)
        predictedResult = ann.predict(np.array(inputs, np.float32), verbose=0)
        # neophodno je da u K-Means algoritam bude prosledjena matrica u kojoj vrste odredjuju elemente
        distances = np.array(distances).reshape(len(distances), 1)
        if max(distances) > 4 * min(distances):
            k_means = KMeans(n_clusters=2, n_init=10)
            k_means.fit(distances)
            currentRes = display_result_with_spaces(predictedResult, alphabet, k_means)
        else:
            currentRes = display_result_as_word(predictedResult, alphabet)
        
        errorSum += hamming_distance(results[i-1], currentRes)
        print(f'{captchaNames[i-1]}-{results[i-1]}-{currentRes}')

    print(errorSum)
