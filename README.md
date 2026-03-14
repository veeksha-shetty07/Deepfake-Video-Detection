# Deepfake Video Detection using Machine Learning

## Project Overview
This project detects deepfake videos using machine learning techniques. 
The system extracts frames from videos, detects faces, and analyzes them 
to identify whether the video is real or manipulated.

## Project Workflow
1. Extract frames from input video
2. Detect faces from the frames
3. Train a machine learning model using the extracted faces
4. Predict whether the video is real or fake

## How to Run the Project

1. Extract frames from video
python extract_frames.py

2. Extract faces from frames
python extract_faces.py

3. Train the model
python train_model.py

4. Predict whether image/video is real or fake
python predict.py

## Files in the Project
extract_frames.py – Extracts frames from videos  
extract_faces.py – Detects and extracts faces from frames  
train_model.py – Trains the deepfake detection model  
predict.py – Predicts whether a video is real or fake  

## Technologies Used
Python  
OpenCV  
PyTorch  
Machine Learning  

## Note
The dataset, extracted frames, and trained model are not uploaded to this repository due to large file size.
