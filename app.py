from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
import io
import os
import time
from AudioGen import AudioGen  # Assume your AudioGen class is saved as AudioGen.py

app = Flask(__name__)
audio_gen = AudioGen()

# Ensure the generated_audio folder exists
os.makedirs('generated_audio', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/random-image', methods=['GET'])
def random_image():
    """
    API endpoint to select a random image and generate white noise audio.
    """
    # Select random image
    image = audio_gen.select_random_image()

    # Convert PIL image to bytes for response
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Generate white noise and save with a unique filename
    timestamp = int(time.time() * 1000)
    filename = f"output_{timestamp}.wav"
    filepath = os.path.join('generated_audio', filename)
    audio = audio_gen.generate_white_noise(image)
    audio.export(filepath, format="wav")

    # Return image and audio filename
    return jsonify({
        'message': 'Random image and audio generated',
        'audio_url': f'/play-audio?filename={filename}',
        'image_data': img_byte_arr.getvalue().decode('latin1')  # Convert image bytes to string for JSON
    })

@app.route('/upload-image', methods=['POST'])
def upload_image():
    """
    API endpoint to upload an image and generate white noise audio.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    uploaded_file = request.files['image']
    image = Image.open(uploaded_file.stream)

    # Convert PIL image to bytes for response
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Generate white noise and save with a unique filename
    timestamp = int(time.time() * 1000)
    filename = f"output_{timestamp}.wav"
    filepath = os.path.join('generated_audio', filename)
    audio = audio_gen.generate_white_noise(image)
    audio.export(filepath, format="wav")

    # Return image and audio filename
    return jsonify({
        'message': 'Image uploaded and audio generated',
        'audio_url': f'/play-audio?filename={filename}',
        'image_data': img_byte_arr.getvalue().decode('latin1')  # Convert image bytes to string for JSON
    })
    
@app.route('/play-audio', methods=['GET'])
def play_audio():
    """
    Serve the generated audio file.
    """
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    filepath = os.path.join('generated_audio', filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    return send_file(filepath, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(debug=True)
