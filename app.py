from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from keras.config import enable_unsafe_deserialization
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from skimage.metrics import structural_similarity as ssim
from os.path import splitext
from flask import request, redirect, url_for
import subprocess
from flask import send_from_directory



enable_unsafe_deserialization()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


###############################################################################################################

# for transformer:

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.pos_encoding.shape[1],
            'd_model': self.pos_encoding.shape[-1],
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


###############################################################################################################
FRAME_HEIGHT, FRAME_WIDTH = 64, 64
NUM_INPUT_FRAMES = 10
NUM_PREDICT_FRAMES = 5
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 0.01

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_loss(y_true, y_pred):
    return ssim_loss(y_true, y_pred) + tf.reduce_mean(tf.keras.losses.MAE(y_true, y_pred))

class PredRNN(tf.keras.Model):
    def __init__(self, custom_input_shape, num_predict_frames, **kwargs):
        super(PredRNN, self).__init__(**kwargs)
        self.custom_input_shape = custom_input_shape 
        self.num_predict_frames = num_predict_frames
        self.encoder = models.Sequential([
            layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.2)
        ])
        self.decoder = models.Sequential([
            layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
        ])
        self.output_layer = layers.Conv3D(3, (3, 3, 3), padding='same', activation='sigmoid')

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return self.output_layer(decoded[:, :self.num_predict_frames])

    def get_config(self):
        config = super(PredRNN, self).get_config()
        config.update({
            'custom_input_shape': self.custom_input_shape,
            'num_predict_frames': self.num_predict_frames
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(
            custom_input_shape=config.get('custom_input_shape', (10, 64, 64, 3)),
            num_predict_frames=config.get('num_predict_frames', 5)
        )

class VideoDataLoader(Sequence):
    def __init__(self, dataframe, batch_size):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.indices = np.arange(len(self.dataframe))

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_clips = self.dataframe.iloc[batch_indices]
        input_frames, output_frames = [], []

        for _, row in batch_clips.iterrows():
            video_path = row['clip_path']
            cap = cv2.VideoCapture(video_path)
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)) / 255.0
                frames.append(frame)
            cap.release()

            if len(frames) >= NUM_INPUT_FRAMES + NUM_PREDICT_FRAMES:
                input_frame_indices = []
                remaining_frames = len(frames) - NUM_PREDICT_FRAMES
                idx = 0

                while len(input_frame_indices) < NUM_INPUT_FRAMES:
                    input_frame_indices.append(idx)
                    if len(input_frame_indices) % 2 == 1:
                        idx += 6 # my skip 5 logic here
                    else:
                        idx += 1 

                    if idx >= remaining_frames:
                        input_frame_indices = list(range(remaining_frames - NUM_INPUT_FRAMES, remaining_frames))

                input_frames.append([frames[i] for i in input_frame_indices])
                output_frames.append(frames[NUM_INPUT_FRAMES:NUM_INPUT_FRAMES + NUM_PREDICT_FRAMES])

        return np.array(input_frames), np.array(output_frames)

custom_input_shape = (NUM_INPUT_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3)
model = PredRNN(custom_input_shape, NUM_PREDICT_FRAMES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=combined_loss, metrics=['mae'])

##########################################################################################################################

loaded_models = {
    "lstm": None,
    "rnn": None,
    "transformer": None  
}


def load_models():
    global loaded_models
    enable_unsafe_deserialization() 
    
    if not loaded_models["lstm"]:
        loaded_models["lstm"] = load_model("models/conv_lstm_model.keras", custom_objects={
            'combined_loss': lambda y_true, y_pred: 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)) +
                                                    tf.reduce_mean(tf.keras.losses.MAE(y_true, y_pred)),
        })
    if not loaded_models["rnn"]:
        loaded_models["rnn"] = load_model("models/PredRNN.keras", custom_objects={
            'PredRNN': PredRNN, 
            'combined_loss': combined_loss,  
            'ssim_loss': ssim_loss 
        })

    if not loaded_models["transformer"]:
        loaded_models["transformer"] =  load_model("models/tmodel.keras", custom_objects={
    'PositionalEncoding': PositionalEncoding,
    'combined_loss': combined_loss,
    'ssim_loss': ssim_loss
        })

@app.route('/')
def index():
  
    return render_template('index.html', models=loaded_models.keys())


UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload', methods=['POST'])



@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files or 'model' not in request.form:
        return "No video file or model selected", 400

    selected_model = request.form['model']
    if selected_model not in loaded_models or loaded_models[selected_model] is None:
        return f"Model '{selected_model}' is not loaded or invalid", 400

    video = request.files['video']
    if video.filename == '':
        return "No selected video file", 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    video_base_name = splitext(video.filename)[0]
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_base_name)
    os.makedirs(output_folder, exist_ok=True)

    try:
        input_frames, predicted_frames = process_video(video_path, selected_model, output_folder)

        predicted_video_path = os.path.join(output_folder, "predicted_video.mp4")
        save_video(predicted_frames, predicted_video_path)

        return render_template(
            'results.html',
            video_name=video.filename,
            video_base_name=video_base_name,
            output_folder=output_folder,
            selected_model=selected_model
        )
    except Exception as e:
        return f"Error processing video: {e}", 500





def generate_frames_and_predictions(video_path, video_base_name):
    output_dir = os.path.join(OUTPUT_FOLDER, video_base_name)
    os.makedirs(output_dir, exist_ok=True)

    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    input_frames = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"input_frame_{frame_count + 1}.png")
        cv2.imwrite(frame_path, frame)
        input_frames.append(frame)
        frame_count += 1
        if frame_count >= 10: 
            break

    video_capture.release()

    predicted_frames = []
    for i, frame in enumerate(input_frames[:5]): 
        predicted_frame = frame 
        predicted_frame_path = os.path.join(output_dir, f"predicted_frame_{i + 1}.png")
        cv2.imwrite(predicted_frame_path, predicted_frame)
        predicted_frames.append(predicted_frame)

    original_video_path = os.path.join(output_dir, "original_video.mp4")
    save_video_from_frames(input_frames, original_video_path)

    predicted_video_path = os.path.join(output_dir, "predicted_video.mp4")
    save_video_from_frames(predicted_frames, predicted_video_path)

    return output_dir

def save_video_from_frames(frames, video_path, fps=10, duration_in_seconds=15):
  
    if not frames:
        return

    height, width, _ = frames[0].shape

    total_frames = int(fps * duration_in_seconds)

    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

  
    frame_count = 0
    while frame_count < total_frames:
        for frame in frames:
            if frame_count >= total_frames: 
                break
            video_writer.write(frame)
            frame_count += 1

    video_writer.release()




@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


def process_video(video_path, model_name, output_folder):
    model = loaded_models[model_name]
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64)) / 255.0
        frames.append(frame)
    cap.release()

    if len(frames) < 15:
        raise ValueError("Video does not have enough frames for prediction!")

    input_frames = np.expand_dims(frames[:10], axis=0)  
    predicted_batch = model.predict(input_frames)
    predicted_frames = predicted_batch[0]

    for i, frame in enumerate(frames[:10]):
        plt.imsave(os.path.join(output_folder, f"input_frame_{i+1}.png"), frame)

    for i, frame in enumerate(predicted_frames):
        plt.imsave(os.path.join(output_folder, f"predicted_frame_{i+1}.png"), frame)

    return frames[:10], predicted_frames



@app.route('/uploads/<filename>')
def serve_uploaded_video(filename):
    """
    Serve uploaded videos from the 'uploads' directory.
    """
    return send_from_directory('uploads', filename)




def save_video(frames, output_path):
    clip = ImageSequenceClip([np.uint8(frame * 255) for frame in frames], fps=10)  
    clip.write_videofile(output_path, codec="libx264")




@app.route('/results')
def results():
    
    video_base_name = "example_video"
    return render_template('results.html', video_base_name=video_base_name)



if __name__ == '__main__':
    load_models() 
    app.run(debug=True)
