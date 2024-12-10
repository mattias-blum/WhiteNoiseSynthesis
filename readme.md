---

# Randomized Image-to-Audio Synthesis with CLIP and Pydub

This project uses OpenAI's CLIP model and environmental audio analysis tools to generate environmental audio (like rain, birds chirping, sea waves, etc.) from random or user-provided images. The system leverages similarity computation with CLIP embeddings and synthesizes environmental audio using `pydub`.

---

## **Table of Contents**

1. [Introduction](#introduction)  
2. [Installation Instructions](#installation-instructions)  
3. [Running the Web App](#running-the-web-app)  
4. [AudioGen Class Documentation](#audiogen-class-documentation)  
5. [Flask App API Calls Explained](#flask-app-api-calls-explained)  
6. [Dependencies](#dependencies)  

---

## **Introduction**

The goal of this project is to integrate images with audio synthesis by:
1. Using CLIP (Contrastive Language–Image Pre-training) embeddings to find similarities between uploaded images or random images and predefined environmental sound categories.
2. Mapping these semantic concepts (identified via similarity scores) to corresponding sound categories.
3. Using `pydub` to generate audio by combining sounds corresponding to these categories.

The pipeline includes:
- Tokenization of audio categories.
- Semantic similarity computation using CLIP embeddings.
- Audio processing with `pydub`.
- A simple web application powered by Flask that allows user interaction.

---

## **Installation Instructions**

Follow these step-by-step instructions to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/random-image-to-audio.git
cd random-image-to-audio
```

---

### 2. Install Dependencies
Install all required dependencies with the following command:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
After installation, start the web application server:
```bash
python app.py
```

The application will be available at [http://localhost:5000](http://localhost:5000).

---

## **Running the Web App**

Once the server is running (`python app.py`), you can interact with the application by:

1. Navigating to `http://localhost:5000` in your web browser.
2. Uploading your own image or allowing the server to randomly select an image.
3. Playing the corresponding synthesized environmental sound generated based on the image.

---

## **AudioGen Class Documentation**

The **AudioGen** class is the core logic of this project, managing CLIP embeddings, similarity computations, and audio generation.

### **Class: `AudioGen`**
Responsible for:
- Encoding images with CLIP embeddings.
- Mapping images to environmental sound categories.
- Generating audio using the most semantically similar categories.

---

### **Methods**

#### `__init__(self)`
- **Purpose**: Initializes the CLIP model, loads pre-trained embeddings, and prepares environmental audio metadata.
- **Dependencies Loaded**:
  - CLIP model from OpenAI.
  - Audio categories from ESC-50.

#### `select_random_image(self)`
- **Purpose**: Selects a random image from the preloaded image dataset.
- **Returns**: The randomly selected image.

---

#### `generate_white_noise(self, image)`
- **Inputs**: An image (randomly selected or user-uploaded).
- **Process**:
  1. Encodes the input image using CLIP embeddings.
  2. Computes cosine similarity between the image embeddings and audio categories.
  3. Selects the top 3 categories with the highest similarity.
  4. Generates audio using corresponding environmental sounds from the ESC-50 dataset.
- **Returns**: Audio data (in `pydub.AudioSegment`) representing synthesized environmental noise.

---

## **Flask App API Calls Explained**

The web server is implemented in **app.py**, powered by Flask. Below is a summary of the available API calls:

---

### **1. `/random_image`**
- **Method**: `GET`
- **Description**: Randomly selects an image from the preloaded image set.
- **Response**: JSON response with the image URL for the front end to display.

---

### **2. `/upload_image`**
- **Method**: `POST`
- **Description**: Accepts a user-uploaded image for processing.
- **Inputs**: An image uploaded by the user via the frontend.
- **Response**: Similarity computation performed, and the top environmental sounds are returned.

---

### **3. `/generate_audio`**
- **Method**: `GET`
- **Description**: Generates and returns audio based on computed similarities.
- **Response**: The audio data (in `.wav` format) for playback on the frontend.

---

## **Dependencies**

The full list of dependencies is stored in **requirements.txt**:

### **`requirements.txt`**
```plaintext
flask==3.0.0
torch==2.0.0
clip-by-openai==1.0
pydub==0.25.1
pandas==2.0.3
numpy==1.23.1
```

---

## **Acknowledgements**
This project builds on the following tools and datasets:
1. OpenAI's CLIP model for semantic similarity computation.
2. The ESC-50 environmental sound dataset for audio categories.
3. Flask and Pydub for web server implementation and audio manipulation.

---