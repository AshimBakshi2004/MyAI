from flask import Flask, request, render_template, jsonify
import easyocr
import numpy as np
import cv2
import io

app = Flask(__name__)

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'])  # You can add other languages like ['en', 'es', 'fr']

@app.route('/process', methods=['POST'])
def process_image():
    try:
        file = request.files['image']
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Convert BGR to RGB for EasyOCR
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use EasyOCR to extract text
        result = reader.readtext(rgb_img)

        # Extract only the text (without confidence scores)
        extracted_text = '\n'.join([text for (bbox, text, confidence) in result])

        if extracted_text.strip():
            return jsonify({
                'extracted_text': extracted_text.strip(),
                'status': 'success'
            })
        else:
            return jsonify({
                'extracted_text': 'No text found in image.',
                'status': 'no_text'
            })
    
    except Exception as e:
        return jsonify({
            'error': f'Error processing image: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
