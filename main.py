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
if not path.exists(settings.REPO_PATH):
    print('Downloading First Order Model package...')
    makedirs(settings.REPO_PATH)
    repo = Repo.clone_from(settings.REPO_URL, settings.REPO_PATH)

if not path.exists(settings.MODELS_PATH):
    makedirs(settings.MODELS_PATH)

if not path.exists(settings.OUTPUTS_PATH):
    makedirs(settings.OUTPUTS_PATH)

# Add First Order Model to the Python path
sys.path.insert(1, './' + settings.REPO_PATH)
from demo import load_checkpoints, make_animation

def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani


# Load video and source image
source_image = imageio.imread(settings.SOURCE_IMAGE_PATH)
driving_video = imageio.mimread(settings.DRIVING_VIDEO_PATH)

reader = imageio.get_reader(settings.DRIVING_VIDEO_PATH)
driving_video_FPS = reader.get_meta_data()['fps']

source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
driving_video_audio = editor.AudioFileClip(settings.DRIVING_VIDEO_PATH)

# Show source image and loaded video
# HTML( display(source_image, driving_video).to_html5_ghnvideo() )


# Create a model and load checkpoints
generator, kp_detector = load_checkpoints(config_path=path.join(settings.CONFIG_PATH, settings.CONFIG_FILE_NAME), checkpoint_path=path.join(settings.MODELS_PATH, settings.MODEL_FILE))

# Perform image animation
predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

# Save resulting video, video can be downloaded from /outputs folder
generated_file_name = settings.MODEL_NAME + 'video-generated.mp4'
generated_file_path = path.join(settings.OUTPUTS_PATH, generated_file_name)
imageio.mimsave(generated_file_path, [img_as_ubyte(frame) for frame in predictions], fps=driving_video_FPS)

print("Setting video audio....")
generated_audio_file_name = path.join(settings.OUTPUTS_PATH,settings.MODEL_NAME+'video-generated.mp4')
final_video = editor.VideoFileClip(generated_file_path)
final_clip = final_video.set_audio(driving_video_audio)
try:
    final_clip.write_videofile(generated_audio_file_name, fps=driving_video_FPS, codec='libx264')
except:
    print('No audio')
# HTML(display(source_image, driving_video, predictions).to_html5_video())