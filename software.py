from typing import Union
from fastapi import FastAPI,  HTTPException, Request
from elevenlabs import clone, generate, play, save
from elevenlabs import set_api_key
import os
import transformers
from transformers import pipeline
import openai
import whisper
import torchaudio
import base64
import json
from tqdm import tqdm
from skimage import img_as_ubyte
import torch
from scipy.spatial import ConvexHull

import warnings
from IPython.display import HTML, clear_output
from skimage.transform import resize
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import imageio
import face_alignment


p = pipeline("automatic-speech-recognition")
openai.api_key = "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"
set_api_key("1473cf6c7f740868fcdc82052b85bbb8")
######################################################################333333333333##############################


def face():
  %cd /content
  !git clone --depth 1 https://github.com/eyaler/first-order-model
  !wget -O ./first-order-model/vox-adv-cpk.pth.tar --no-check-certificate -nc https://openavatarify.s3.amazonaws.com/weights/vox-adv-cpk.pth.tar
  !wget -O ./first-order-model/vox-adv-cpk.pth.tar --no-check-certificate -nc https://eyalgruss.com/fomm/vox-adv-cpk.pth.tar

  !mkdir -p /root/.cache/torch/hub/checkpoints
  %cd /root/.cache/torch/hub/checkpoints
  !wget --no-check-certificate -nc https://eyalgruss.com/fomm/s3fd-619a316812.pth
  !wget --no-check-certificate -nc https://eyalgruss.com/fomm/2DFAN4-11f355bf06.pth.tar
  %cd /content
  !pip install -U git+https://github.com/ytdl-org/youtube-dl
  !pip install imageio==2.9.0
  !pip install imageio-ffmpeg==0.4.5
  !pip install git+https://github.com/1adrianb/face-alignment@v1.0.1
  !pip install pyyaml==5.4.1

  start_seconds = 0
  duration_seconds =  60
  start_seconds = max(start_seconds,0)
  duration_seconds = max(duration_seconds,0)
  if duration_seconds:
    !mv /content/uploaded_video/video_1.mp4 /content/full_video
    !ffmpeg -ss $start_seconds -t $duration_seconds -i /content/full_video -f mp4 /content/uploaded_video/video_1.mp4 -y
  center_video_to_head = True
  crop_video_to_head = True
  video_crop_expansion_factor = 2.5
  center_image_to_head = True
  crop_image_to_head = False
  image_crop_expansion_factor = 2.5
  video_crop_expansion_factor = max(video_crop_expansion_factor, 1)
  image_crop_expansion_factor = max(image_crop_expansion_factor, 1)

  import imageio
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.animation as animation
  from skimage.transform import resize
  from IPython.display import HTML, clear_output
  import warnings
  warnings.filterwarnings("ignore")
  %cd first-order-model
  import face_alignment
  fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                        device='cuda')

  def create_bounding_box(target_landmarks, expansion_factor=1):
      target_landmarks = np.array(target_landmarks)
      x_y_min = target_landmarks.reshape(-1, 68, 2).min(axis=1)
      x_y_max = target_landmarks.reshape(-1, 68, 2).max(axis=1)
      expansion_factor = (expansion_factor-1)/2
      bb_expansion_x = (x_y_max[:, 0] - x_y_min[:, 0]) * expansion_factor
      bb_expansion_y = (x_y_max[:, 1] - x_y_min[:, 1]) * expansion_factor
      x_y_min[:, 0] -= bb_expansion_x
      x_y_max[:, 0] += bb_expansion_x
      x_y_min[:, 1] -= bb_expansion_y
      x_y_max[:, 1] += bb_expansion_y
      return np.hstack((x_y_min, x_y_max-x_y_min))

  def fix_dims(im):
      if im.ndim == 2:
          im = np.tile(im[..., None], [1, 1, 3])
      return im[...,:3]

  def get_crop(im, center_face=True, crop_face=True, expansion_factor=1, landmarks=None):
      im = fix_dims(im)
      if (center_face or crop_face) and not landmarks:
          landmarks = fa.get_landmarks_from_image(im)
      if (center_face or crop_face) and landmarks:
          rects = create_bounding_box(landmarks, expansion_factor=expansion_factor)
          x0,y0,w,h = sorted(rects, key=lambda x: x[2]*x[3])[-1]
          if crop_face:
              s = max(h, w)
              x0 += (w-s)//2
              x1 = x0 + s
              y0 += (h-s)//2
              y1 = y0 + s
          else:
              img_h,img_w = im.shape[:2]
              img_s = min(img_h,img_w)
              x0 = min(max(0, x0+(w-img_s)//2), img_w-img_s)
              x1 = x0 + img_s
              y0 = min(max(0, y0+(h-img_s)//2), img_h-img_s)
              y1 = y0 + img_s
      else:
          h,w = im.shape[:2]
          s = min(h,w)
          x0 = (w-s)//2
          x1 = x0 + s
          y0 = (h-s)//2
          y1 = y0 + s
      return int(x0),int(x1),int(y0),int(y1)

  def pad_crop_resize(im, x0=None, x1=None, y0=None, y1=None, new_h=256, new_w=256):
      im = fix_dims(im)
      h,w = im.shape[:2]
      if x0 is None:
        x0 = 0
      if x1 is None:
        x1 = w
      if y0 is None:
        y0 = 0
      if y1 is None:
        y1 = h
      if x0<0 or x1>w or y0<0 or y1>h:
          im = np.pad(im, pad_width=[(max(-y0,0),max(y1-h,0)),(max(-x0,0),max(x1-w,0)),(0,0)], mode='edge')
      return resize(im[max(y0,0):y1-min(y0,0),max(x0,0):x1-min(x0,0)], (new_h, new_w))

  source_image = imageio.imread('/content/uploaded_picture/image_1.jpg')
  source_image = pad_crop_resize(source_image, *get_crop(source_image, center_face=center_image_to_head, crop_face=crop_image_to_head, expansion_factor=image_crop_expansion_factor))

  with imageio.get_reader('/content/uploaded_video/video_1.mp4', format='mp4') as reader:
    fps = reader.get_meta_data()['fps']

    driving_video = []
    landmarks = None
    try:
        for i,im in enumerate(reader):
            if not crop_video_to_head:
                break
            landmarks = fa.get_landmarks_from_image(im)
            if landmarks:
                break
        x0,x1,y0,y1 = get_crop(im, center_face=center_video_to_head, crop_face=crop_video_to_head, expansion_factor=video_crop_expansion_factor, landmarks=landmarks)
        reader.set_image_index(0)
        for im in reader:
            driving_video.append(pad_crop_resize(im,x0,x1,y0,y1))
    except RuntimeError:
        pass
  #@title Find best alignment

  %cd /content/first-order-model
  from demo import load_checkpoints
  generator, kp_detector = load_checkpoints(config_path='/content/first-order-model/config/vox-adv-256.yaml',
                              checkpoint_path='/content/first-order-model/vox-adv-cpk.pth.tar')

  from scipy.spatial import ConvexHull
  def normalize_kp(kps):
      max_area = 0
      max_kp = None
      for kp in kps:
          kp = kp - kp.mean(axis=0, keepdims=True)
          area = ConvexHull(kp[:, :2]).volume
          area = np.sqrt(area)
          kp[:, :2] = kp[:, :2] / area
          if area>max_area:
            max_area = area
            max_kp = kp
      return max_kp

  from tqdm import tqdm

  kp_source = fa.get_landmarks_from_image(255 * source_image)
  if kp_source:
    norm_kp_source = normalize_kp(kp_source)

  norm  = float('inf')
  best = 0
  best_kp_driving = None
  for i, image in tqdm(enumerate(driving_video)):
    kp_driving = fa.get_landmarks_from_image(255 * image)
    if kp_driving:
      norm_kp_driving = normalize_kp(kp_driving)
      if kp_source:
        new_norm = (np.abs(norm_kp_source - norm_kp_driving) ** 2).sum()
        if new_norm < norm:
          norm = new_norm
          best = i
          best_kp_driving = kp_driving
      else:
        best_kp_driving = kp_driving
        break

  exaggerate_factor = 1
  adapt_movement_scale = True
  use_relative_movement = True
  use_relative_jacobian = True

  import torch
  from skimage import img_as_ubyte

  def full_normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                  use_relative_movement=False, use_relative_jacobian=False, exaggerate_factor=1):
      if adapt_movement_scale:
          source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
          driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
          adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
      else:
          adapt_movement_scale = 1

      kp_new = {k: v for k, v in kp_driving.items()}

      if use_relative_movement:
          kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
          kp_value_diff *= adapt_movement_scale * exaggerate_factor
          kp_new['value'] = kp_value_diff + kp_source['value']

          if use_relative_jacobian:
              jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
              kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

      return kp_new

  def make_animation(source_image, driving_video, generator, kp_detector, adapt_movement_scale=False,
                  use_relative_movement=False, use_relative_jacobian=False, cpu=False, exaggerate_factor=1):
      with torch.no_grad():
          predictions = []
          source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
          if not cpu:
              source = source.cuda()
          driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
          kp_source = kp_detector(source)
          kp_driving_initial = kp_detector(driving[:, :, 0])

          for frame_idx in tqdm(range(driving.shape[2])):
              driving_frame = driving[:, :, frame_idx]
              if not cpu:
                  driving_frame = driving_frame.cuda()
              kp_driving = kp_detector(driving_frame)
              kp_norm = full_normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial, adapt_movement_scale=adapt_movement_scale, use_relative_movement=use_relative_movement,
                                    use_relative_jacobian=use_relative_jacobian, exaggerate_factor=exaggerate_factor)
              out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

              predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
      return predictions

  predictions_forward = make_animation(source_image, driving_video[best:], generator, kp_detector, adapt_movement_scale=adapt_movement_scale, use_relative_movement=use_relative_movement,
                                    use_relative_jacobian=use_relative_jacobian, exaggerate_factor=exaggerate_factor)
  predictions_backward = make_animation(source_image, driving_video[:(best+1)][::-1], generator, kp_detector, adapt_movement_scale=adapt_movement_scale, use_relative_movement=use_relative_movement,
                                    use_relative_jacobian=use_relative_jacobian, exaggerate_factor=exaggerate_factor)

  imageio.mimsave('/content/generated.mp4', [img_as_ubyte(frame) for frame in predictions_backward[::-1] + predictions_forward[1:]], fps=fps)
  #!ffmpeg -i /content/generated.mp4 -i /content/video -c:v libx264 -c:a aac -map 0:v -map 1:a? -pix_fmt yuv420p /content/final.mp4 -profile:v baseline -movflags +faststart -y
  #video can be downloaded from /content/final.mp4
  %cd /content/
  return {"message": "Files uploaded successfully"}
################################################################################################################
messages = [
    {"role": "system", "content": "You are an AI assisstant, your task is to understand the user will say and respond according to that. After understanding, cinsider the user input as a prompt format."},
]
def chatbot(text):
    if text:
        messages.append({"role": "user", "content": text})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        return reply

model = whisper.load_model("medium")

def transcribe(audio):

    audio_read, _ = torchaudio.load(audio)
    torchaudio.save('/content/streaming.wav', audio_read, sample_rate=22050)
    #load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    text = result.text
    response = chatbot(text)
    return response

#####################################################################################################################################
def retalk():
  from IPython.display import clear_output, display, HTML

  #display(HTML('<font color="red">Cloning video-retalking repository:</font>'))
  !git clone https://github.com/justinjohn0306/video-retalking
  %cd video-retalking
  #clear_output()

  display(HTML('<font color="red">Uninstalling existing gdown and reinstalling from source to avoid Google Drive download quota issues:</font>'))
  !pip uninstall gdown -y
  !pip install git+https://github.com/wkentaro/gdown.git
  #clear_output()

  display(HTML('<font color="red">Installing other project requirements:</font>'))
  !pip install -r requirements_colab.txt
  #clear_output()
  #display(HTML('<font color="red">Now we are set up and ready to proceed!</font>'))
  from IPython.display import clear_output, display, HTML
  #%cd /content/video-retalking

  import gdown

  gdown.download("https://drive.google.com/uc?id=1Qtg-GVUKZ7aXtz-4O9Mm4ncXjEYRB8-p", "/content/checkpoints.zip", quiet=False)
  !unzip -o /content/checkpoints.zip -d /content/video-retalking/
  #!rm /content/checkpoints.zip
  #clear_output()

  #display(HTML('<font color="red">Now you are ready to run the inference!</font>'))
  import os
  from IPython.display import clear_output, display, HTML


  face_video = '/content/generated.mp4'
  audio_input = '/content/output_voice.mp3'
  output_file = '/content/1_1.mp4'

  assert os.path.exists(face_video), f"Face video file not found: {face_video}"
  assert os.path.exists(audio_input), f"Audio input file not found: {audio_input}"

  display(HTML('<font color="red">Running the inference...</font>'))
  !python3 inference.py --face $face_video --audio $audio_input --outfile $output_file
  %cd /content/
  return
######################################################################################################################################

### Isolate audio, video and pictues from api call
def custom_split_function(byte_stream: bytes) -> tuple:
    data = json.loads(byte_stream)

    audio_data = data.get("audio", "")
    image_data = data.get("image", "")
    video_data = data.get("video", "")

    audio_bytes = base64.b64decode(audio_data)
    image_bytes = base64.b64decode(image_data)
    video_bytes = base64.b64decode(video_data)

    return audio_bytes, image_bytes, video_bytes

###################################################################################################################################

app = FastAPI()

SAVE_PATH_AUDIO = "uploaded_audio"
SAVE_PATH_IMAGE = "uploaded_picture"
SAVE_PATH_VIDEO = "uploaded_video"

@app.post("/upload/")
async def upload_files(request: Request):
    byte_stream = await request.body()

    audio_bytes, image_bytes, video_bytes = custom_split_function(byte_stream)

    os.makedirs(SAVE_PATH_AUDIO, exist_ok=True)
    os.makedirs(SAVE_PATH_IMAGE, exist_ok=True)
    os.makedirs(SAVE_PATH_VIDEO, exist_ok=True)

    audio_filename = f"voice_{len(os.listdir(SAVE_PATH_AUDIO)) + 1}.mp3"
    audio_path = os.path.join(SAVE_PATH_AUDIO, audio_filename)
    with open(audio_path, 'wb') as buffer:
        buffer.write(audio_bytes)

    image_filename = f"image_{len(os.listdir(SAVE_PATH_IMAGE)) + 1}.jpg"
    image_path = os.path.join(SAVE_PATH_IMAGE, image_filename)
    with open(image_path, 'wb') as buffer:
        buffer.write(image_bytes)

    video_filename = f"video_{len(os.listdir(SAVE_PATH_VIDEO)) + 1}.mp4"  # Adjust the video format as needed
    video_path = os.path.join(SAVE_PATH_VIDEO, video_filename)
    with open(video_path, 'wb') as buffer:
        buffer.write(video_bytes)

    # Your existing transcription and cloning logic here using the saved `audio_path`, `image_path`, and `video_path`
    res = transcribe(audio_path)

    voice = clone(
        name="clone",
        description="An old American male voice with a slight hoarseness in his throat. Perfect for news",
        files=[audio_path],
    )

    audio = generate(text=res, voice=voice)
    output_mp3_filename = "output_voice.mp3"
    save(audio,output_mp3_filename)
    face()
    retalk()
    return {"message": "Files uploaded successfully"}


