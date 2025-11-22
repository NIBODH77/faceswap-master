#!/usr/bin/env python3
"""
Web frontend for FaceSwap
Allows users to upload images/videos and perform face swaps through a browser interface
"""
import os
import sys
import logging
import time
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import subprocess
import uuid
import shutil
import glob
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MODELS_FOLDER'] = 'models'

# Allowed extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Training status tracker
training_sessions = {}

# Create necessary folders
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['MODELS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, 'temp'), exist_ok=True)

def allowed_file(filename, file_type='image'):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads for source and target"""
    try:
        if 'source' not in request.files or 'target' not in request.files:
            return jsonify({'error': 'Source and target files required'}), 400

        source_file = request.files['source']
        target_file = request.files['target']

        if source_file.filename == '' or target_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400

        # Determine file types
        source_type = 'video' if allowed_file(source_file.filename, 'video') else 'image'
        target_type = 'video' if allowed_file(target_file.filename, 'video') else 'image'

        if not (allowed_file(source_file.filename, source_type) and 
                allowed_file(target_file.filename, target_type)):
            return jsonify({'error': 'Invalid file type'}), 400

        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Save files
        source_filename = secure_filename(source_file.filename)
        target_filename = secure_filename(target_file.filename)

        source_path = os.path.join(session_dir, 'source_' + source_filename)
        target_path = os.path.join(session_dir, 'target_' + target_filename)

        source_file.save(source_path)
        target_file.save(target_path)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'source_type': source_type,
            'target_type': target_type,
            'source_path': source_path,
            'target_path': target_path
        })

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract_faces():
    """Extract faces from uploaded images or videos"""
    try:
        data = request.json
        session_id = data.get('session_id')
        file_type = data.get('file_type', 'source')

        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400

        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)

        if not os.path.exists(session_dir):
            return jsonify({'error': 'Session not found'}), 404

        pattern = os.path.join(session_dir, f'{file_type}_*')
        all_matches = glob.glob(pattern)
        files = [f for f in all_matches if os.path.isfile(f) and not f.endswith('_faces')]

        if not files:
            return jsonify({'error': f'{file_type.capitalize()} file not found'}), 404

        input_file = files[0]
        output_dir = os.path.join(session_dir, f'{file_type}_faces')
        os.makedirs(output_dir, exist_ok=True)

        # Enhanced extraction with better settings
        cmd = [
            sys.executable, 'faceswap.py', 'extract',
            '-i', input_file,
            '-o', output_dir,
            '-D', 's3fd',
            '-A', 'fan',
            '-M', 'bisenet-fp',
            '-nm', 'hist'
        ]

        logger.info(f"Running extraction: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Extraction failed"
            logger.error(f"Extraction failed: {error_msg}")
            return jsonify({'error': 'Face extraction failed', 'details': error_msg}), 500

        face_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        face_count = len(face_files)

        if face_count == 0:
            return jsonify({'error': 'No faces detected'}), 400

        return jsonify({
            'success': True,
            'faces_extracted': face_count,
            'output_dir': output_dir
        })

    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/convert', methods=['POST'])
def convert_faces():
    """Convert faces using trained model"""
    try:
        data = request.json
        session_id = data.get('session_id')
        use_trained_model = data.get('use_trained_model', True)

        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400

        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        os.makedirs(output_dir, exist_ok=True)

        # Find target file
        target_pattern = os.path.join(session_dir, 'target_*')
        target_files = [f for f in glob.glob(target_pattern) if os.path.isfile(f)]

        if not target_files:
            return jsonify({'error': 'Target file not found'}), 404

        target_file = target_files[0]

        # Determine which model to use
        if use_trained_model:
            model_dir = os.path.join(app.config['MODELS_FOLDER'], 'trained_model')
            if not os.path.exists(model_dir):
                logger.warning("Trained model not found, using default instant swap")
                model_dir = None
        else:
            model_dir = None

        # Run conversion
        if model_dir and os.path.exists(model_dir):
            # Use trained model for better results
            cmd = [
                sys.executable, 'faceswap.py', 'convert',
                '-i', target_file,
                '-o', output_dir,
                '-m', model_dir,
                '-c', 'avg-color',
                '-M', 'extended',
                '-w', 'opencv'
            ]
        else:
            # InstantSwap fallback (no training needed but lower quality)
            logger.info("Using instant swap (no trained model)")
            source_faces = os.path.join(session_dir, 'source_faces')
            if not os.path.exists(source_faces):
                return jsonify({'error': 'Source faces not extracted'}), 400

            # Simple copy and swap logic
            cmd = [
                sys.executable, 'faceswap.py', 'convert',
                '-i', target_file,
                '-o', output_dir,
                '-al', source_faces,
                '-w', 'opencv'
            ]

        logger.info(f"Running conversion: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Conversion failed"
            logger.error(f"Conversion failed: {error_msg}")
            return jsonify({'error': 'Face conversion failed', 'details': error_msg}), 500

        output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]

        if not output_files:
            return jsonify({'error': 'No output generated'}), 500

        output_file = output_files[0]

        return jsonify({
            'success': True,
            'output_url': url_for('download_result', session_id=session_id, filename=output_file),
            'filename': output_file,
            'used_trained_model': model_dir is not None
        })

    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<session_id>/<filename>')
def download_result(session_id, filename):
    """Download the converted result"""
    try:
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        file_path = os.path.join(output_dir, filename)

        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        return send_file(file_path, as_attachment=True)

    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("FaceSwap Web Application")
    print("=" * 60)
    print("\nबेहतर results के लिए:")
    print("1. पहले train_dataset_builder.py run करें")
    print("2. Proper dataset के साथ model train करें")
    print("3. फिर web app में trained model का उपयोग करें")
    print("\n" + "=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)