from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from scripts.equation_processing import HandwrittenEquationProcessor
import matplotlib
matplotlib.use("Agg") 
import re

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'heic'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.h5')
label_encoder_path = os.path.join(os.path.dirname(__file__), 'models', 'label_encoder.pkl')
# Then use model_path instead of 'models/model.h5'

# Model paths
MODEL_PATH = model_path
LABEL_ENCODER_PATH = label_encoder_path

# Global processor instance
processor = None

def initialize_processor():
    """Initialize the processor once at startup"""
    global processor
    
    if processor is not None:
        return True
    
    try:
        processor = HandwrittenEquationProcessor(MODEL_PATH, LABEL_ENCODER_PATH)
        return True
    except Exception as e:
        print(f"Error initializing processor: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_numerical_result(calculation_result):
    """Extract numerical result from calculation string"""
    try:
        if '=' in calculation_result:
            parts = calculation_result.split('=')
            if len(parts) >= 2:
                result_part = parts[-1].strip()
                numbers = re.findall(r'-?\d+\.?\d*', result_part)
                if numbers:
                    number_str = numbers[-1]
                    return float(number_str) if '.' in number_str else int(number_str)
        return None
    except:
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if processor is initialized
        global processor
        if processor is None:
            initialize_processor()

        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file'}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Get image info
        from PIL import Image
        with Image.open(file_path) as img:
            image_info = {
                'width': img.width,
                'height': img.height,
                'format': img.format
            }

        # Process the equation
        processing_result = processor.process_and_calculate(file_path, show_visualization=True)
        
        # Extract results
        calculation_result = processing_result['result']
        numerical_result = extract_numerical_result(calculation_result)

        return jsonify({
            'success': True,
            'filename': filename,
            'image_info': image_info,
            'equation': processing_result['recognized'],
            'result': numerical_result if numerical_result is not None else calculation_result,
            'full_calculation': calculation_result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing equation: {str(e)}'
        }), 500

if __name__ == '__main__':
    initialize_processor()
    app.run(debug=True, port=5001)