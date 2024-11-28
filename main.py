import cv2
import random
from datetime import datetime
from scipy.stats import mode
import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # To plot the images

from sign_detection.sign_detector import SignDetector
from neural_network.src.recognize_image import Recog
from template_match.template_matcher import TemplateMatcher
from sign_translation.sign_translator import SignTranslator

translator = SignTranslator()

def process_image(image, methods=["nn", "tm", "sift"]):
    '''
    Processes the input image to find the sign.

    @param input_file: the name of the input image.

    @return the results of the sign detection.
    '''

    signs = SignDetector().find_signs(image.copy())

    results = []

    list_a = []
    list_b = []
    list_c = []
    if len(signs) == 0:
        top_choice, top_choice_list, score = match(image, methods)
        print(translator.get_sign(top_choice.mode))

        if top_choice.count != 0:
            list_a.append(top_choice.mode)
            list_b.append(top_choice_list)
            list_c.append(score)
    for sign in signs:
        cv2.imwrite("test.jpg", sign[0])
        top_choice, top_choice_list, score = match(sign[0], methods)

        if top_choice.count != 0:
            sign_details = dict()
            sign_details["x"] = sign[1][0]
            sign_details["y"] = sign[1][1]
            sign_details["w"] = sign[1][2]
            sign_details["h"] = sign[1][3]

            results.append(sign_details)
            list_a.append(top_choice.mode)
            list_b.append(top_choice_list)
            list_c.append(score)

    for idx, result in enumerate(results):
        r = translator.get_sign(list_a[idx])
        print(r)
        sign = result
        if len(results):
            cv2.rectangle(image, (sign["x"], sign["y"]), (sign["x"] + sign["w"], sign["y"] + sign["h"]), (0, 255, 0), 2)
            cv2.putText(image, r, (sign["x"], sign["y"]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return list_a, list_b, list_c, image


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
    if mode(test_results).count == 1:
        test_results[2] = test_results[0]

    top_choice = mode(test_results)

    return top_choice, test_results, guesses[0][0]


def nn_match(sign):
    '''
    Uses the neural network to perform a detection on the sign.

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


def process_folder(input_folder):
    '''
    Process all images in a folder and visualize results in a plot.

    @param input_folder: the path to the folder containing the images.
    '''
    images = [f for f in os.listdir(input_folder) if f.endswith(('jpg', 'png', 'jpeg'))]
    
    # Calculate number of rows and columns for subplot grid based on number of images
    num_images = len(images)
    import math
    cols = int(math.sqrt(num_images))  # Number of columns per row (can adjust)
    rows = (num_images + cols - 1) // cols  # Calculate number of rows needed
    
    # Create subplots with dynamic size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust figure size as needed
    axes = axes.flatten()  # Flatten to make it iterable in case of multiple rows

    for i, image_name in enumerate(images):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        top_choice, top_choice_list, score, result_image = process_image(image)

        # Convert image to RGB for plotting
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_image_rgb = cv2.resize(result_image_rgb, (300, 300))

        label = translator.get_sign(top_choice[0])

        axes[i].imshow(result_image_rgb)
        axes[i].set_title(f"{image_name}\n{label}", fontsize=10)  # Display image name and label
        axes[i].axis('off')  # Hide axis

    # Turn off axes for any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    input_folder = "inputs"  # Replace with the path to your folder containing images
    process_folder(input_folder)

    # inputs = "./inputs/bien bao cam.jpg"
    # image = cv2.imread(inputs)
    # top_choice, top_choice_list, score, result_image = process_image(image)
    # print(top_choice)
    # print(top_choice_list)
    # print(score)
    
