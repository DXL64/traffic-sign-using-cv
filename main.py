import cv2
import random
from datetime import datetime
from scipy.stats import mode
import os
import json
import tensorflow as tf
import numpy as np

from sign_detection.sign_detector import SignDetector
from neural_network.src.recognize_image import Recog
from template_match.template_matcher import TemplateMatcher
from sign_translation.sign_translator import SignTranslator

INPUT_DIR = "inputs/"
OUTPUT_DIR = "results/"


def process_image(image, methods=["nn", "tm", "sift"]):
    '''
    Processes the input image to find the sign.

    @param input_file: the name of the input image.

    @return the results of the sign detection.
    '''
    random.seed(datetime.now().timestamp())

    signs = SignDetector().find_signs(image.copy())

    translator = SignTranslator()

    results = dict()

    for sign in signs:
        top_choice, top_choice_list = match(sign[0], methods)


        if top_choice.count != 0:
            sign_details = dict()
            sign_details["x"] = sign[1][0]
            sign_details["y"] = sign[1][1]
            sign_details["w"] = sign[1][2]
            sign_details["h"] = sign[1][3]

            results[translator.get_sign(top_choice.mode)] = sign_details

    for r in results.keys():
        sign = results[r]
        cv2.rectangle(image, (sign["x"], sign["y"]), (sign["x"] + sign["w"], sign["y"] + sign["h"]), (0, 255, 0), 2)
        cv2.putText(image, r, (sign["x"], sign["y"]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return top_choice_list, results


def match(sign, methods):
    test_results = []
    guesses = []


    if "nn" in methods:
        res, guesses = nn_match(sign)
        test_results.append(res)

    if "tm" in methods:
        test_results.append(template_match(sign, guesses))

    if "sift" in methods:
        test_results.append(sift_match(sign, guesses))

    top_choice = mode(test_results)

    return top_choice, test_results


def nn_match(sign):
    '''
    Uses the neual network to perform a detection on the sign.

    @param sign: the sign to analyze.

    @returns the top detection.
    '''
    model_path = os.path.join('models', 'best_model.h5')
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    image = cv2.resize(sign, (30, 30), interpolation=cv2.INTER_NEAREST)
    image_array = np.array(image) / 1.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict the class
    prediction = model.predict(image_array)
    
    # Get top 5 predictions with their probabilities
    top_indices = np.argsort(prediction[0])[-5:][::-1]  # Get indices of top 5 predictions
    top_probabilities = prediction[0][top_indices]     # Get corresponding probabilities

    # Create list of tuples with (class_index, probability)
    top_recogs = [(float(prob), int(idx)) for idx, prob in zip(top_indices, top_probabilities)]
    if len(top_recogs) > 0:
        return top_recogs[0][1], top_recogs
    return -1, []


def template_match(sign, guesses=[]):
    '''
    Uses the template matching to perform a detection on the sign.

    @param sign: the sign to analyze.

    @returns the top detection.
    '''
    if sign.shape[2] == 3:  # Check if the image is not already grayscale
        sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)

    top_recogs = TemplateMatcher().template_match(sign, guesses)
    if len(top_recogs) > 0:
        return top_recogs[0][1]

    return -1


def sift_match(sign, guesses=[]):
    '''
    Uses the sift matching to perform a detection on the sign.

    @param sign: the sign to analyze.

    @returns the top detection.
    '''
    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)

    top_recogs = TemplateMatcher().sift_match(sign, guesses)
    if len(top_recogs) > 0:
        return top_recogs[0][1]

    return -1


if __name__ == '__main__':
    image = cv2.imread(INPUT_DIR + "test2.jpeg")
    top_choice, top_choice_list = process_image(image)
    print(top_choice)
    print(top_choice_list)