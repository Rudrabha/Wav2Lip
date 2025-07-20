from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import subprocess
from werkzeug.utils import secure_filename
# from sepratemp3 import *


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
# ALLOWED_EXTENSIONS = {'txt', 'mp3', 'mp4', 'wav', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'mp3', 'mp4', 'wav', 'webm'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def generate_lip_sync(video_output_path):
    # Replace this command with your actual Wav2Lip command
    checkpoint_path = r'D:\\flask\Wav2Lip\\checkpoints\\wav2lip_gan.pth'
    face_path = os.path.join(app.config["UPLOAD_FOLDER"], "input.mp4")
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "input.mp3")
    
    wav2lip_command = f'python D:\\flask\\Wav2Lip\\inference.py --checkpoint_path "{checkpoint_path}" --face "{face_path}" --audio "{audio_path}" --outfile "{video_output_path}"'
    subprocess.run(wav2lip_command, shell=True)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/process', methods=['POST'])
def process():
    text = request.form['text']

    # Generate lip-synced video
    video_output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
    generate_lip_sync(video_output_path)

    return redirect(url_for('result'))
@app.route('/result')
def result():
    return render_template('result.html')
@app.route('/output/<filename>')
def get_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
@app.route('/download', methods=['GET', 'POST'])
def download():
    # import pdb;pdb.set_trace()
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], 'useroutput.mp4')
    return send_from_directory(app.config['OUTPUT_FOLDER'], 'useroutput.mp4', as_attachment=True)
 
 
@app.route('/execute_sepratemp3', methods=['POST'])
def execute_sepratemp3():
    try:
        # Assuming your sepratemp3 script is in the same directory as your app.py
        script_path = os.path.join(os.getcwd(), 'sepratemp3.py')

        # Execute sepratemp3 script
        subprocess.run(['python', script_path])

        return jsonify({'message': 'sepratemp3 executed successfully'})
    except Exception as e:
        return jsonify({'error': f'Error executing sepratemp3: {str(e)}'})



if __name__ == '__main__':
    app.run(debug=True)
