# VoxSent
## Description

This project implements an Emotion Analysis System that utilizes deep learning models trained on the GoEmotions dataset. The project includes two versions of emotion classification
- 28-Emotion Model: Predicts one of 28 emotions (with a neutral class included).
- 7-Emotion Model: Predicts one of 7 primary emotions (anger, disgust, neutral, surprise, joy, sadness, fear).

The trained models are integrated into a Streamlit-based web application that allows users to analyze emotions from text input and speech input (transcribed using Whisper).

## Dataset
The models are trained on the [GoEmotions Dataset on Kaggle](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset), which is annotated with multiple emotions per text. Data preprocessing includes:
* Cleaning text
* Tokenization
* Padding sequences
* Data augmentation (for the 7-emotion model)

## Installation
```
# Create a new conda environment
conda create --name emo python=3.10
conda activate emo

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg
conda install -c conda-forge ffmpeg

# Run the Streamlit app
streamlit run main.py
```

## Troubleshooting

**Problem**: Emotion classification is a challenging task, particularly when dealing with imbalanced datasets. The 28-emotion model suffers from data imbalance

**Solution**: Trained 7-emotion model with balanced data and augmentation.

## Usage
1. **Run the Streamlit app** to analyze emotions from text and speech.
2. **Upload an audio file or type text** to generate predictions.
3. **Select the emotion model** 28-emotion or 7-emotion.
4. **Receive detailed analysis** with predicted emotions and emoji-based visualization.
5. **Explore training insights** with confusion matrices and model evaluation reports in about me page.

## Model Architecture
### 28-Emotion Model
(img)
### 7-Emotion Model
(img)

## Results
### Training History
#### 28-Emotion Model
(img)
#### 7-Emotion Model
(img)

### Confusion Matrix
#### 28-Emotion Model
(img)
#### 7-Emotion Model
(img)

### Classification Report
#### 28-Emotion Model
(img)
#### 7-Emotion Model
(img)
