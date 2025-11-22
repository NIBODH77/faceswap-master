
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
            return jsonify({'error': 'Session not found. Please upload files first.'}), 404
        
        # Find the actual uploaded file (not directory) BEFORE creating output_dir
        pattern = os.path.join(session_dir, f'{file_type}_*')
        all_matches = glob.glob(pattern)
        # Filter to get only files, not directories
        files = [f for f in all_matches if os.path.isfile(f) and not f.endswith('_faces')]
        
        if not files:
            return jsonify({'error': f'{file_type.capitalize()} file not found. Please upload files first.'}), 404
        
        input_file = files[0]
        
        # Validate file exists and is readable
        if not os.path.exists(input_file) or not os.path.isfile(input_file):
            return jsonify({'error': f'Invalid {file_type} file path'}), 400
        
        # Check file size
        file_size = os.path.getsize(input_file)
        if file_size == 0:
            return jsonify({'error': f'{file_type.capitalize()} file is empty'}), 400
        
        logger.info(f"Extracting faces from {file_type} file: {input_file} (size: {file_size} bytes)")
        
        # Now create output directory after finding the input file
        output_dir = os.path.join(session_dir, f'{file_type}_faces')
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine if input is video or image based on extension
        file_ext = input_file.lower().split('.')[-1]
        is_video = file_ext in ['mp4', 'avi', 'mov', 'mkv']
        
        # For single images, we need to use a directory as input in faceswap
        # Create a temp directory with just this image
        if not is_video:
            temp_input_dir = os.path.join(session_dir, f'{file_type}_temp_input')
            os.makedirs(temp_input_dir, exist_ok=True)
            # Copy image to temp directory
            temp_image_path = os.path.join(temp_input_dir, os.path.basename(input_file))
            shutil.copy2(input_file, temp_image_path)
            extraction_input = temp_input_dir
        else:
            extraction_input = input_file
        
        # Run extraction
        cmd = [
            sys.executable, 'faceswap.py', 'extract',
            '-i', extraction_input,
            '-o', output_dir,
            '-D', 's3fd',
            '-A', 'fan'
        ]
        
        logger.info(f"Running extraction command: {' '.join(cmd)}")
        # Capture both stdout and stderr. Extraction can take time (downloading models, processing)
        # First run might take 10+ minutes for model downloads
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown extraction error"
            logger.error(f"Extraction failed for {file_type}: {error_msg}")
            logger.error(f"Full output: {result.stdout}")
            
            # Provide more helpful error messages
            if "not a valid video" in error_msg.lower():
                return jsonify({
                    'error': 'File format issue detected',
                    'details': 'The file may be corrupted or in an unsupported format. Please try a different image (PNG, JPG, JPEG, BMP) or video (MP4, AVI, MOV, MKV) file.'
                }), 500
            
            return jsonify({'error': 'Face extraction failed', 'details': error_msg}), 500
        
        # Count extracted faces
        try:
            face_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            face_count = len(face_files)
            logger.info(f"Successfully extracted {face_count} faces from {file_type}")
        except Exception as e:
            logger.error(f"Error counting faces: {str(e)}")
            face_count = 0
        
        if face_count == 0:
            # Cleanup temp directory if it was created
            if not is_video:
                temp_input_dir = os.path.join(session_dir, f'{file_type}_temp_input')
                if os.path.exists(temp_input_dir):
                    shutil.rmtree(temp_input_dir)
            
            return jsonify({
                'error': 'No faces detected',
                'details': f'No faces were found in the {file_type} file. Please ensure the image/video contains clear, visible faces.'
            }), 400
        
        # Cleanup temp directory if it was created
        if not is_video:
            temp_input_dir = os.path.join(session_dir, f'{file_type}_temp_input')
            if os.path.exists(temp_input_dir):
                shutil.rmtree(temp_input_dir)
        
        return jsonify({
            'success': True,
            'faces_extracted': face_count,
            'output_dir': output_dir,
            'message': f'Successfully extracted {face_count} face(s) from {file_type}'
        })
    
    except subprocess.TimeoutExpired:
        logger.error("Extraction timed out")
        return jsonify({
            'error': 'Extraction took too long',
            'details': 'The extraction process timed out. This may happen with very large video files. Please try with a shorter video or smaller images.'
        }), 500
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return jsonify({'error': 'Unexpected error during extraction', 'details': str(e)}), 500

def run_training_background(session_id, cmd):
    """Run training in background and update status"""
    try:
        training_sessions[session_id] = {'status': 'training', 'progress': 0}
        logger.info(f"Starting training: {' '.join(cmd)}")
        
        # Training can take significant time - use long timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            training_sessions[session_id] = {'status': 'completed', 'progress': 100}
            logger.info(f"Training completed for session {session_id}")
        else:
            error_msg = result.stderr or result.stdout or "Unknown training error"
            logger.error(f"Training failed for session {session_id}: {error_msg}")
            training_sessions[session_id] = {'status': 'failed', 'progress': 0, 'error': error_msg}
            
    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out for session {session_id}")
        training_sessions[session_id] = {'status': 'failed', 'progress': 0, 'error': 'Training timed out'}
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
        
        if not os.path.exists(session_dir):
            return jsonify({'error': 'Session not found. Please upload files first.'}), 404
        
        model_dir = os.path.join(app.config['MODELS_FOLDER'], session_id)
        
        source_faces = os.path.join(session_dir, 'source_faces')
        target_faces = os.path.join(session_dir, 'target_faces')
        
        # Validate that face extraction has been completed
        if not os.path.exists(source_faces):
            return jsonify({
                'error': 'Source faces not extracted',
                'details': 'Please extract faces from the source image/video first.'
            }), 400
        
        if not os.path.exists(target_faces):
            return jsonify({
                'error': 'Target faces not extracted',
                'details': 'Please extract faces from the target image/video first.'
            }), 400
        
        # Count faces in both directories
        source_face_count = len([f for f in os.listdir(source_faces) if f.endswith(('.png', '.jpg', '.jpeg'))])
        target_face_count = len([f for f in os.listdir(target_faces) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if source_face_count == 0:
            return jsonify({
                'error': 'No source faces found',
                'details': 'The source faces directory is empty. Please re-extract faces from the source image/video.'
            }), 400
        
        if target_face_count == 0:
            return jsonify({
                'error': 'No target faces found',
                'details': 'The target faces directory is empty. Please re-extract faces from the target image/video.'
            }), 400
        
        # Check minimum image requirement
        MIN_IMAGES_REQUIRED = 25
        if source_face_count < MIN_IMAGES_REQUIRED or target_face_count < MIN_IMAGES_REQUIRED:
            return jsonify({
                'error': 'Not enough training data',
                'details': f'FaceSwap requires at least {MIN_IMAGES_REQUIRED} images per side for training. You have {source_face_count} source and {target_face_count} target faces. Please use videos or multiple images with more faces (ideally 500-5000 images per side).',
                'source_count': source_face_count,
                'target_count': target_face_count,
                'minimum_required': MIN_IMAGES_REQUIRED
            }), 400
        
        logger.info(f"Starting training with {source_face_count} source faces and {target_face_count} target faces")
        
        # Check if training is already in progress
        current_status = training_sessions.get(session_id, {})
        if current_status.get('status') in ['starting', 'training']:
            return jsonify({
                'error': 'Training already in progress',
                'details': 'Please wait for the current training to complete.'
            }), 400
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Run training with validated parameters
        cmd = [
            sys.executable, 'faceswap.py', 'train',
            '-A', source_faces,
            '-B', target_faces,
            '-m', model_dir,
            '-t', 'original',
            '-b', '16',
            '-i', str(iterations),  # Updated from -it to -i
            '-s', '100',
            '-I', '1000',  # Updated from -ss to -I (snapshot-interval)
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
            'message': f'Training started with {source_face_count} source and {target_face_count} target faces',
            'model_dir': model_dir,
            'session_id': session_id,
            'iterations': iterations
        })
    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': 'Unexpected error starting training', 'details': str(e)}), 500

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
        current_status = training_status_data.get('status', 'not_started')
        
        if current_status != 'completed':
            if current_status == 'failed':
                error_details = training_status_data.get('error', 'Unknown error')
                return jsonify({
                    'error': 'Training failed',
                    'details': f'Training did not complete successfully: {error_details}'
                }), 400
            elif current_status in ['starting', 'training']:
                return jsonify({
                    'error': 'Training in progress',
                    'details': 'Please wait for training to complete before converting.'
                }), 400
            else:
                return jsonify({
                    'error': 'Training not started',
                    'details': 'Please train the model first before converting.'
                }), 400
        
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        model_dir = os.path.join(app.config['MODELS_FOLDER'], session_id)
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        
        # Validate session directory exists
        if not os.path.exists(session_dir):
            return jsonify({'error': 'Session not found'}), 404
        
        # Check if model directory and files exist
        if not os.path.exists(model_dir):
            return jsonify({
                'error': 'Model directory not found',
                'details': 'Training may not have completed successfully. Please try training again.'
            }), 404
        
        model_files = glob.glob(os.path.join(model_dir, '*.h5'))
        if not model_files:
            return jsonify({
                'error': 'Model file not found',
                'details': 'No trained model files (.h5) found. Training may not have completed successfully. Please check training status and try again.'
            }), 404
        
        logger.info(f"Found {len(model_files)} model file(s) for session {session_id}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find target file (filter for actual files, not directories)
        target_pattern = os.path.join(session_dir, 'target_*')
        all_target_matches = glob.glob(target_pattern)
        # Get only files that are not directories and don't end with _faces
        target_files = [f for f in all_target_matches if os.path.isfile(f) and not f.endswith('_faces')]
        
        if not target_files:
            return jsonify({
                'error': 'Target file not found',
                'details': 'The target image/video file is missing. Please upload files again.'
            }), 404
        
        target_file = target_files[0]
        
        # Validate target file exists and is readable
        if not os.path.exists(target_file) or not os.path.isfile(target_file):
            return jsonify({'error': 'Invalid target file'}), 400
        
        logger.info(f"Converting target file: {target_file}")
        
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
        
        logger.info(f"Running conversion command: {' '.join(cmd)}")
        # Capture both stdout and stderr. Conversion can take time depending on media size
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown conversion error"
            logger.error(f"Conversion failed: {error_msg}")
            logger.error(f"Full output: {result.stdout}")
            return jsonify({
                'error': 'Face conversion failed',
                'details': error_msg
            }), 500
        
        logger.info("Conversion completed successfully")
        
        # Find output file
        try:
            output_files = [f for f in os.listdir(output_dir) if not f.startswith('.') and os.path.isfile(os.path.join(output_dir, f))]
        except Exception as e:
            logger.error(f"Error listing output directory: {str(e)}")
            output_files = []
        
        if not output_files:
            return jsonify({
                'error': 'No output generated',
                'details': 'Conversion completed but no output file was created. This may happen if no faces were detected in the target image/video.'
            }), 500
        
        output_file = output_files[0]
        logger.info(f"Successfully created output file: {output_file}")
        
        return jsonify({
            'success': True,
            'output_url': url_for('download_result', session_id=session_id, filename=output_file),
            'filename': output_file,
            'message': 'Face swap completed successfully!'
        })
    
    except subprocess.TimeoutExpired:
        logger.error("Conversion timed out")
        return jsonify({
            'error': 'Conversion took too long',
            'details': 'The conversion process timed out. This may happen with very large video files. Please try with a shorter video or smaller images.'
        }), 500
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        return jsonify({'error': 'Unexpected error during conversion', 'details': str(e)}), 500

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
    import warnings
    warnings.filterwarnings('ignore', message='This is a development server')
    
    # Set werkzeug logger to ERROR level to suppress warnings
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
