
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
    """Extract faces from uploaded images"""
    try:
        data = request.json
        session_id = data.get('session_id')
        file_type = data.get('file_type', 'source')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        
        # Find the actual uploaded file (not directory) BEFORE creating output_dir
        pattern = os.path.join(session_dir, f'{file_type}_*')
        all_matches = glob.glob(pattern)
        # Filter to get only files, not directories
        files = [f for f in all_matches if os.path.isfile(f)]
        
        if not files:
            return jsonify({'error': 'File not found'}), 404
        
        input_file = files[0]
        
        # Now create output directory after finding the input file
        output_dir = os.path.join(session_dir, f'{file_type}_faces')
        os.makedirs(output_dir, exist_ok=True)
        
        # Run extraction
        cmd = [
            sys.executable, 'faceswap.py', 'extract',
            '-i', input_file,
            '-o', output_dir,
            '-D', 's3fd',
            '-A', 'fan',
            '-M', 'bisenet-fp'
        ]
        
        logger.info(f"Running extraction: {' '.join(cmd)}")
        # Capture both stdout and stderr
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown extraction error"
            logger.error(f"Extraction failed: {error_msg}")
            return jsonify({'error': 'Extraction failed', 'details': error_msg}), 500
        
        # Count extracted faces
        try:
            face_count = len([f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        except:
            face_count = 0
        
        return jsonify({
            'success': True,
            'faces_extracted': face_count,
            'output_dir': output_dir
        })
    
    except subprocess.TimeoutExpired:
        logger.error("Extraction timed out")
        return jsonify({'error': 'Extraction timed out'}), 500
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def run_training_background(session_id, cmd):
    """Run training in background and update status"""
    try:
        training_sessions[session_id] = {'status': 'training', 'progress': 0}
        logger.info(f"Starting training: {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Monitor training output
        for line in process.stdout:
            logger.info(f"Training: {line.strip()}")
            # Update progress based on output
            if '[' in line and ']' in line:
                try:
                    # Extract iteration progress
                    parts = line.split('[')[1].split(']')[0].split('/')
                    if len(parts) == 2:
                        current = int(parts[0].strip())
                        total = int(parts[1].strip())
                        progress = int((current / total) * 100)
                        training_sessions[session_id]['progress'] = progress
                except:
                    pass
        
        process.wait()
        
        if process.returncode == 0:
            training_sessions[session_id] = {'status': 'completed', 'progress': 100}
            logger.info(f"Training completed for session {session_id}")
        else:
            training_sessions[session_id] = {'status': 'failed', 'progress': 0}
            logger.error(f"Training failed for session {session_id}")
            
    except Exception as e:
        logger.error(f"Training background error: {str(e)}")
        training_sessions[session_id] = {'status': 'failed', 'progress': 0, 'error': str(e)}

@app.route('/train', methods=['POST'])
def train_model():
    """Train a face swap model"""
    try:
        data = request.json
        session_id = data.get('session_id')
        iterations = data.get('iterations', 5000)
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        model_dir = os.path.join(app.config['MODELS_FOLDER'], session_id)
        
        source_faces = os.path.join(session_dir, 'source_faces')
        target_faces = os.path.join(session_dir, 'target_faces')
        
        if not os.path.exists(source_faces) or not os.path.exists(target_faces):
            return jsonify({'error': 'Please extract faces first'}), 400
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Run training
        cmd = [
            sys.executable, 'faceswap.py', 'train',
            '-A', source_faces,
            '-B', target_faces,
            '-m', model_dir,
            '-t', 'original',
            '-bs', '16',
            '-it', str(iterations),
            '-s', '100',
            '-ss', '1000',
            '-ps', '100'
        ]
        
        # Initialize training status BEFORE starting thread to avoid transient "not_found"
        training_sessions[session_id] = {'status': 'starting', 'progress': 0}
        
        # Start training in background thread
        thread = threading.Thread(target=run_training_background, args=(session_id, cmd))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started',
            'model_dir': model_dir,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/training_status/<session_id>', methods=['GET'])
def training_status(session_id):
    """Check training status"""
    status = training_sessions.get(session_id, {'status': 'not_found', 'progress': 0})
    return jsonify(status)

@app.route('/convert', methods=['POST'])
def convert_faces():
    """Convert faces using trained model"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Check if training is complete
        training_status_data = training_sessions.get(session_id, {})
        if training_status_data.get('status') != 'completed':
            current_status = training_status_data.get('status', 'not_started')
            return jsonify({
                'error': f'Training not completed. Current status: {current_status}. Please wait for training to complete.'
            }), 400
        
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        model_dir = os.path.join(app.config['MODELS_FOLDER'], session_id)
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        
        # Check if model exists
        model_files = glob.glob(os.path.join(model_dir, '*.h5'))
        if not model_files:
            return jsonify({'error': 'Model file not found. Training may not have completed successfully.'}), 404
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find target file (filter for actual files, not directories)
        target_pattern = os.path.join(session_dir, 'target_*')
        all_target_matches = glob.glob(target_pattern)
        # Get only files that are not directories and don't end with _faces
        target_files = [f for f in all_target_matches if os.path.isfile(f) and not f.endswith('_faces')]
        
        if not target_files:
            return jsonify({'error': 'Target file not found'}), 404
        
        target_file = target_files[0]
        
        # Run conversion
        cmd = [
            sys.executable, 'faceswap.py', 'convert',
            '-i', target_file,
            '-o', output_dir,
            '-m', model_dir,
            '-c', 'avg-color',
            '-M', 'extended',
            '-w', 'ffmpeg' if target_file.endswith(('.mp4', '.avi', '.mov', '.mkv')) else 'opencv'
        ]
        
        logger.info(f"Running conversion: {' '.join(cmd)}")
        # Capture both stdout and stderr
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown conversion error"
            logger.error(f"Conversion failed: {error_msg}")
            return jsonify({'error': 'Conversion failed', 'details': error_msg}), 500
        
        # Find output file
        output_files = [f for f in os.listdir(output_dir) if not f.startswith('.')]
        if not output_files:
            return jsonify({'error': 'No output generated. Check if target image contains faces.'}), 500
        
        output_file = output_files[0]
        
        return jsonify({
            'success': True,
            'output_url': url_for('download_result', session_id=session_id, filename=output_file),
            'filename': output_file
        })
    
    except subprocess.TimeoutExpired:
        logger.error("Conversion timed out")
        return jsonify({'error': 'Conversion timed out'}), 500
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

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """Clean up session files"""
    try:
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
