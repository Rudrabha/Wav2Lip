from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import subprocess
from werkzeug.utils import secure_filename
from sepratemp3 import *
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
 
 
@app.route('/execute', methods=['GET','POST'])
def execute_sepratemp3():
    try:
        video_path ='D:\\flask\\Wav2Lip\\output1.mp4'
        extract_spoken_text(video_path)
        extract_audio_and_process(video_path)
        # allfunc(extract_audio_and_process(),extract_spoken_text())
        return result     
    except Exception as e:
        return jsonify({'error': f'Error executing sepratemp3: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)








# -----------------------------------------------------------------------------------------------------------------------
#  from flask import Flask, render_template, request, redirect, url_for, send_from_directory
# import os
# import pyttsx3
# import subprocess

# app = Flask(__name__)
# # import pdb;pdb.set_trace()
# UPLOAD_FOLDER = 'uploads'
# OUTPUT_FOLDER = 'output'
# ALLOWED_EXTENSIONS = {'txt', 'mp3','mp4'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def text_to_speech(text, output_path):
#     engine = pyttsx3.init()
#     engine.save_to_file(text, output_path)
#     engine.runAndWait()

# def generate_lip_sync(video_output_path):
#     # Replace this command with your actual Wav2Lip command
#     checkpoint_path = r'D:\\flask\Wav2Lip\\checkpoints\\wav2lip_gan.pth'
#     face_path = os.path.join(app.config["UPLOAD_FOLDER"], "input.mp4")
#     audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "input.mp3")
    
#     wav2lip_command = f'python D:\\flask\\Wav2Lip\\inference.py --checkpoint_path "{checkpoint_path}" --face "{face_path}" --audio "{audio_path}" --outfile "{video_output_path}"'
#     subprocess.run(wav2lip_command, shell=True)


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/process', methods=['POST'])
# def process():
#     text = request.form['text']

#     # Save text to speech
#     mp3_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.mp3')
#     text_to_speech(text, mp3_path)

#     # Generate lip-synced video
#     video_output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
#     generate_lip_sync(video_output_path)

#     return redirect(url_for('result'))

# @app.route('/result')
# def result():
#     return render_template('result.html')

# # Add this route to serve the generated video
# @app.route('/output/<filename>')
# def get_output(filename):
#     return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# @app.route('/download')
# def download():
#     video_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
#     return send_from_directory(app.config['OUTPUT_FOLDER'], 'output.mp4', as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True)