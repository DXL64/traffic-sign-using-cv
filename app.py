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

def run():
    model_dir = "models"
    
    print("----------- Starting Streamlit App -----------")
    st.title("Ứng dụng nhận diện biển báo giao thông")
    
    list_labels = []
    with open(os.path.join(model_dir, 'labels.txt'), 'r') as f:
        for line in f:
            list_labels.append(line)

    # Option to upload an image file, provide a URL, or capture with camera
    st.write("Lựa chọn 1 phương thức để nhập ảnh đầu vào:")
    option = st.radio("Chọn đầu vào", ('Tải ảnh biển báo', 'Nhập đường dẫn URL', 'Sử dụng webcam'))

    image = None
    if option == 'Tải ảnh biển báo':
        uploaded_file = st.file_uploader("Tải ảnh biển báo", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            # Read the uploaded image file in OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_tmp = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = image_tmp
    
    elif option == 'Nhập đường dẫn URL':
        image_url = st.text_input("Nhập đường dẫn URL vào đây:")
        if image_url:
            image_tmp = load_image_from_url(image_url)
            image = image_tmp
    
    elif option == 'Sử dụng webcam':
        image_file = st.camera_input("Chụp ảnh từ webcam")
        if image_file is not None:
            # Convert the captured image to an OpenCV format
            image_bytes = np.asarray(bytearray(image_file.getvalue()), dtype=np.uint8)
            image_tmp = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            image = image_tmp
    
    # Predict if an image was loaded
    if image is not None:
        # Preprocess the image
        top_choice, top_choice_list, score, image = process_image(image)
        
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Ảnh biển báo giao thông', use_column_width=True)

        for i in range(len(top_choice)):
            st.write(f"Dự đoán lớp: {top_choice[i]}")
            st.write(f"Dự đoán nhãn: {list_labels[top_choice[i]]}")
            st.write(f"Độ tin cậy: {score}")
            st.write(f"Nhãn từ mạng Neural: {list_labels[top_choice_list[i][0]]}")
            st.write(f"Nhãn từ Template Matching: {list_labels[top_choice_list[i][1]]}")
            st.write(f"Nhãn từ Sift Matching: {list_labels[top_choice_list[i][2]]}")
    
    print("----------- Streamlit App Running -----------")

if __name__ == "__main__":
    run()
