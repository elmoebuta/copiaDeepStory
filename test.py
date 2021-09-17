from os import path, makedirs
import imageio
import numpy as np
import sys
import uuid
from moviepy import editor
import settings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
from skimage import img_as_ubyte
from ctypes import cdll
from git import Repo
warnings.filterwarnings("ignore")

# Check if First Order Model package has not been cloned
if not path.exists('first_order_model'):
    print('Downloading First Order Model package...')
    makedirs('first_order_model')
    repo = Repo.clone_from('https://github.com/AliaksandrSiarohin/first-order-model', 'first-order-model')

if not path.exists('models'):
    makedirs('models')

if not path.exists('webs/static/files/generated'):
    makedirs('web/static/files/generated')

# Add First Order Model to the Python path
sys.path.insert(1, '../first_order_model')
from demo import load_checkpoints, make_animation

D_VIDEO_PATH = ''
S_IMAGE_PATH = ''

# Load video and source image
source_image = imageio.imread(S_IMAGE_PATH)
driving_video = imageio.mimread(D_VIDEO_PATH)

reader = imageio.get_reader(D_VIDEO_PATH)
driving_video_FPS = reader.get_meta_data()['fps']

source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
driving_video_audio = editor.AudioFileClip(D_VIDEO_PATH)
    
# Create a model and load checkpoints
generator, kp_detector = load_checkpoints(config_path='first-order-model/config/vox-adv-256.yaml', checkpoint_path='first-order-model/config/vox-adv-cpk.pth.tar')

# Perform image animation
predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

# Save resulting video, video can be downloaded from /outputs folder
generated_file_name = 'vox-adv-video-generated.mp4'
generated_file_path = path.join('web/static/files/generated', generated_file_name)
imageio.mimsave(generated_file_path, [img_as_ubyte(frame) for frame in predictions], fps=driving_video_FPS)

print("Setting video audio....")
generated_audio_file_name = 'web/static/files/generated/vox-adv-video-audio-generated.mp4'
final_video = editor.VideoFileClip(generated_file_path)
final_clip = final_video.set_audio(driving_video_audio)
final_clip.write_videofile(generated_audio_file_name, fps=driving_video_FPS, codec='libx264')