# Human Emotion Detection using CNN

This project is an end-to-end deep learning pipeline for facial emotion recognition using the FER2013 dataset. It uses TensorFlow and Keras to train a Convolutional Neural Network (CNN) that classifies grayscale face images into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Project Structure

```text
human-emotion/
|-- dataset/
|   |-- train/
|   |   |-- angry/
|   |   |-- disgust/
|   |   |-- fear/
|   |   |-- happy/
|   |   |-- sad/
|   |   |-- surprise/
|   |   `-- neutral/
|   |-- validation/
|   |   `-- ...
|   `-- test/
|       `-- ...
|-- models/
|-- outputs/
|   |-- plots/
|   `-- reports/
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- model.py
|   `-- utils.py
|-- app.py
|-- train.py
|-- prepare_dataset.py
|-- predict.py
|-- webcam.py
|-- requirements.txt
`-- README.md
```

## Dataset Download

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download and prepare the FER2013 dataset automatically:

```bash
python prepare_dataset.py --force
```

This downloads the Kaggle dataset `msambare/fer2013`, stores it under the project `dataset/` folder, and creates a validation split from the original training data.

3. The prepared dataset will follow this directory format:

```text
dataset/
|-- train/
|   |-- angry/
|   |-- disgust/
|   |-- fear/
|   |-- happy/
|   |-- sad/
|   |-- surprise/
|   `-- neutral/
|-- validation/
|   |-- angry/
|   |-- disgust/
|   |-- fear/
|   |-- happy/
|   |-- sad/
|   |-- surprise/
|   `-- neutral/
`-- test/
    |-- angry/
    |-- disgust/
    |-- fear/
    |-- happy/
    |-- sad/
    |-- surprise/
    `-- neutral/
```

If you already have the dataset locally, you can also place it into the same structure manually.

## Model Details

- Input image size: `48x48`
- Color mode: grayscale
- Normalization: pixel values scaled to `[0, 1]`
- Augmentation:
  - `rotation_range=10`
  - `width_shift_range=0.1`
  - `height_shift_range=0.1`
  - `horizontal_flip=True`
- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Metrics: `accuracy`
- Epochs: `30`
- Batch size: `64`

## Training

Run the training script:

```bash
python train.py
```

During training, the project will:

- Load images from `dataset/train`, `dataset/validation`, and `dataset/test`
- Train the CNN model
- Save the best checkpoint to `models/best_emotion_model.keras`
- Save the final trained model to `emotion_model.h5`
- Save training plots to `outputs/plots/`
- Save the classification report to `outputs/reports/classification_report.txt`

## Prediction on a Single Image

Use the saved model to predict emotion from one image:

```bash
python predict.py --image path/to/image.jpg
```

Optional custom model path:

```bash
python predict.py --image path/to/image.jpg --model path/to/emotion_model.h5
```

## Streamlit Frontend

Launch the web app:

```bash
streamlit run app.py
```

The Streamlit app includes:

- Image upload emotion detection
- Live webcam emotion detection in the browser
- Face bounding boxes with emotion labels and confidence scores

## Real-Time Webcam Detection with OpenCV

Start desktop webcam-based emotion detection:

```bash
python webcam.py
```

Press `q` to quit the webcam window.

## Evaluation Outputs

After training, the following evaluation artifacts are generated:

- `outputs/plots/training_history.png`
- `outputs/plots/confusion_matrix.png`
- `outputs/reports/classification_report.txt`

## How It Works

The overall project pipeline works in the following stages:

1. Dataset Preparation  
   The FER2013 dataset is downloaded, organized into `train`, `validation`, and `test` folders, and mapped into seven emotion classes.

2. Preprocessing and Augmentation  
   Images are resized to `48x48`, converted to grayscale, normalized to the range `[0, 1]`, and augmented using rotation, shifting, and horizontal flipping to improve generalization.

3. CNN Training  
   A convolutional neural network learns facial-expression features from the training data using stacked convolution, batch normalization, pooling, and dropout layers. The model is trained with Adam optimization and monitored using validation performance.

4. Evaluation  
   After training, the model is tested on unseen images. Accuracy, confusion matrix, and classification report are generated to measure class-wise performance.

5. Model Saving  
   The best-performing model checkpoint is stored during training, and the final trained model is saved as `emotion_model.h5` for reuse in inference applications.

6. Single-Image Inference  
   In `predict.py`, a single input image is loaded, preprocessed in the same way as training data, passed through the trained model, and mapped to the predicted emotion label.

7. Real-Time Detection  
   In `webcam.py` and the Streamlit app, frames from the webcam are captured, faces are detected using Haar cascades, each face is preprocessed, and the trained CNN predicts the corresponding emotion in real time.

### Architecture Summary

```text
FER2013 Dataset
      |
      v
Data Loading + Preprocessing
      |
      v
Image Augmentation
      |
      v
CNN Model Training
      |
      v
Evaluation + Reports
      |
      v
Saved Model (emotion_model.h5)
      |
      +--> predict.py (single image prediction)
      |
      +--> webcam.py (desktop real-time detection)
      |
      `--> app.py (Streamlit web interface)
```

## Resume Project Highlights

- End-to-end CNN-based emotion classification system
- Streamlit frontend with live webcam emotion detection
- Modular training and inference pipeline
- Real-time webcam emotion detection with OpenCV
- Data augmentation, callbacks, evaluation plots, and saved reports

## Project Scope

This project is designed as an end-to-end computer vision application for facial emotion recognition from both static images and live video streams. Its scope includes dataset preparation, preprocessing, CNN-based training, evaluation, saved-model inference, browser-based interaction through Streamlit, and desktop real-time webcam detection with OpenCV.

In its current form, the project is well-suited for:

- Academic mini-projects and final-year deep learning projects
- Resume and portfolio demonstrations
- Prototyping human-centered AI applications
- Basic real-time emotion-aware interface experiments

The project focuses on visible facial-expression classification across seven emotion categories and is intended as a practical deep learning implementation rather than a production-grade affective computing system.

## Applications

Potential use cases for this project include:

- Smart classroom systems to estimate student engagement, confusion, or attention
- Human-computer interaction systems that respond based on user emotion
- Customer feedback and retail analytics for reaction monitoring
- Virtual interview and training simulators
- Healthcare support systems for non-clinical mood observation
- Social robots and intelligent assistants with emotion-aware interaction
- UX testing and market research to analyze emotional response to content or products
- Demo applications for computer vision, AI, and deep learning portfolios

## Future Work

This project can be extended further in several meaningful ways:

- Improve model performance using transfer learning or more advanced CNN architectures
- Add face tracking to improve webcam smoothness and reduce repeated detection overhead
- Introduce model quantization or TensorFlow Lite for faster real-time inference
- Add support for video file emotion analysis in addition to webcam input
- Create a more advanced frontend with custom live camera controls beyond Streamlit defaults
- Add deployment support using Docker, Hugging Face Spaces, or cloud platforms
- Improve robustness for different lighting conditions, head poses, and occlusions
- Add fairness, bias, and privacy considerations for real-world usage
- Save session analytics, prediction logs, or emotion trends over time
- Integrate alerting or adaptive UI behavior based on detected emotion
