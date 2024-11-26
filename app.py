import streamlit as st
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
import requests
import cv2
import os
from main import process_image

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading image: {e}")
        return None

def run(args):
    model_dir = f"models/{args.model_name}"
    model_path = os.path.join(model_dir, 'best_model.h5')
    
    print("----------- Starting Streamlit App -----------")
    st.title("Traffic Sign Recognition App")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    list_labels = []
    with open(os.path.join(model_dir, 'labels.txt'), 'r') as f:
        for line in f:
            list_labels.append(line)

    # Option to upload an image file or provide a URL
    st.write("Choose an option to provide a traffic sign image:")
    option = st.radio("Input Method", ('Upload Image', 'Enter Image URL'))

    image = None
    if option == 'Upload Image':
        uploaded_file = st.file_uploader("Upload a traffic sign image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            # Read the uploaded image file in OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_tmp = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image_tmp is not None:
                image = cv2.resize(image_tmp, (30, 30), interpolation=cv2.INTER_NEAREST)
                st.image(cv2.cvtColor(image_tmp, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_container_width=True)
    
    elif option == 'Enter Image URL':
        image_url = st.text_input("Enter the image URL here:")
        if image_url:
            image_tmp = load_image_from_url(image_url)
            if image_tmp is not None:
                image = cv2.resize(image_tmp, (30, 30), interpolation=cv2.INTER_NEAREST)
                st.image(cv2.cvtColor(image_tmp, cv2.COLOR_BGR2RGB), caption='Image from URL', use_container_width=True)
    
    # Predict if an image was loaded
    if image is not None:
        # Preprocess the image
        top_choice, top_choice_list = process_image(image)
        st.write(f"Predicted Class: {top_choice}")
        st.write(f"Predicted Label: {list_labels[top_choice]}")
        st.write(f"Predicted Score: {top_choice_list[top_choice]}")
    
    print("----------- Streamlit App Running -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlit App for Traffic Sign Recognition")
    parser.add_argument('--model_name', type=str, default='vn', help='Path of the trained model')
    args = parser.parse_args()
    run(args)
