# Traffic Sign Recognition App

A Streamlit web application that classifies uploaded traffic sign images using a trained Keras model.

## Overview

This project demonstrates how a machine learning model can be integrated into a simple web application. Users upload an image of a traffic sign, and the app predicts the most likely traffic sign class using a pre-trained model.

The goal of this project is to practice model deployment, Streamlit app development, and basic machine learning application structure.

## Features

- Upload a traffic sign image
- Run prediction using a trained Keras model
- Display the predicted class
- Use a CSV file for class-label mapping
- Run locally or deploy on Streamlit Community Cloud

## Tech Stack

- **Python**
- **Streamlit**
- **TensorFlow / Keras**
- **Pandas**
- **CSV label mapping**

## Project Structure

```text
traffic-sign-recognition-app/
├── app.py
├── traffic_sign_model.keras
├── labels.csv
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

Make sure you have Python installed.

```bash
python --version
```

### Installation

```bash
git clone https://github.com/Al-Farhan-Seg/traffic-sign-recognition-app.git
cd traffic-sign-recognition-app
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

## Deployment

This app can be deployed on Streamlit Community Cloud.

1. Push the project to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Set `app.py` as the entry point.
5. Select Python 3.11 if TensorFlow dependency issues appear.
6. Deploy the app.

## What I Learned

- How to connect a trained machine learning model to a web interface
- How to use Streamlit for quick ML app deployment
- How to structure files for a small ML project
- How class-label mappings work in image classification apps

## Future Improvements

- [ ] Add confidence scores
- [ ] Improve the UI design
- [ ] Add sample test images
- [ ] Add error handling for invalid uploads
- [ ] Deploy a live version and link it here

## Author

Built by [Farhan Segujja](https://github.com/Al-Farhan-Seg).