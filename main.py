import cv2
import random
from datetime import datetime
from scipy.stats import mode
import os
import json

from sign_detection.sign_detector import SignDetector
from neural_network.src.recognize_image import Recog
from template_match.template_matcher import TemplateMatcher
from sign_translation.sign_translator import SignTranslator

INPUT_DIR = "inputs/"
OUTPUT_DIR = "results/"


def process_image(input_file, output_file="", download=True, methods=["nn", "tm", "sift"]):
    '''
    Processes the input image to find the sign.

    @param input_file: the name of the input image.
    @param output_file: the name of the output image.

    @return the file name for the output image.
    '''
    random.seed(datetime.now().timestamp())

    image = cv2.imread(input_file)

    signs = SignDetector().find_signs(image.copy())

    translator = SignTranslator()

    results = dict()

    for sign in signs:
        top_choice, _ = match(sign[0], methods, sign[2])

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

    if download:
        if output_file == "":
            output_file = "processed_signs_" + str(random.randint(1, 1000))

        output_dir = OUTPUT_DIR + output_file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        output_image = output_dir + "/" + output_file + ".png"
        cv2.imwrite(output_image, image)

        output_file = output_dir + "/" + output_file + ".json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    else:
        output_image = ""
    return output_image, results


def match(sign, methods, shapes = []):
    test_results = []
    guesses = []

    if shapes == []:
        shapes = ["TRIANGLE", "CIRCLE", "SQUARE", "HEXAGON"]

    if "nn" in methods:
        res, guesses = nn_match(sign, shapes)
        test_results.append(res)

    if "tm" in methods:
        test_results.append(template_match(sign, guesses, shapes))

    if "sift" in methods:
        test_results.append(sift_match(sign, guesses, shapes))

    top_choice = mode(test_results)

    print(test_results)

    return top_choice, test_results


def nn_match(sign, shapes = []):
    '''
    Uses the neual network to perform a detection on the sign.

    @param sign: the sign to analyze.

    @returns the top detection.
    '''

    translator = SignTranslator()

    top_recogs = Recog().recog_image(sign)
    new_top_recogs = []
    for recogs in top_recogs:
        if translator.get_shape(recogs[1]) in shapes:
            new_top_recogs.append(recogs)

    if len(new_top_recogs) > 0:
        return new_top_recogs[0][1], new_top_recogs

    return -1, []


def template_match(sign, guesses=[], shapes = []):
    '''
    Uses the template matching to perform a detection on the sign.

    @param sign: the sign to analyze.

    @returns the top detection.
    '''
    if sign.shape[2] == 3:  # Check if the image is not already grayscale
        sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)

    top_recogs = TemplateMatcher().template_match(sign, guesses, shapes)
    if len(top_recogs) > 0:
        return top_recogs[0][1]

    return -1


def sift_match(sign, guesses=[], shapes = []):
    '''
    Uses the sift matching to perform a detection on the sign.

    @param sign: the sign to analyze.

    @returns the top detection.
    '''
    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)

    top_recogs = TemplateMatcher().sift_match(sign, guesses, shapes)
    if len(top_recogs) > 0:
        return top_recogs[0][1]

    return -1


if __name__ == '__main__':
    process_image(INPUT_DIR + "stop.jpg", "result")
