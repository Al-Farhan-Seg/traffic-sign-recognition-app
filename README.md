# Traffic Sign Recognition App

This repository contains a Streamlit app that loads `traffic_sign_model.keras` and predicts the class of an uploaded traffic sign image.

## Files
- `app.py` - Streamlit app entrypoint
- `traffic_sign_model.keras` - trained Keras model
- `labels.csv` - class label mapping
- `requirements.txt` - Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push these files to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repo.
4. Set the entrypoint file to `app.py`.
5. In Advanced settings, choose Python 3.11 if TensorFlow build issues appear.
6. Deploy.
