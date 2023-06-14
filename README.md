# Avatar Animation

This is a student project which allows you to animate the face of a character with your facial expressions and voice. In particular, it contains three ways of animation:
* animation using your facial expressions from the video (it is available in pre-recorded and real-time modes)
* animation using your speech from the audio file (only in offline mode with Russian speech)
* animation using both facial expressions and speech from the videoclip where you show different emotions and pronounce some phrases 


## Installation
The project was developed and tested on Windows 10 using Python 3.9 and packet manager pip.

First of all, you need to install PyTorch framework. In order to do that you should follow the [link](https://pytorch.org/get-started/locally/) and choose parameters of installation, in particular:
* PyTorch Build - Stable
* Your OS - Windows
* Package - Pip
* Language - Python
* Compute Platform - CUDA

It will generate the command line which can be used then for installation.

Then you need to install the remaining project dependencies via command:

```bash
pip install -r requirements.txt
```

## Usage

In the script main.py you can find implementations of four functions that use animation classes with pretty simple interface in order to generate animation. You can call one of them at the end of the script to animate the character in a desired way. 

### Video animation

#### Real-time mode
```python
video_animation_online()  # this will start video animation in real-time
```
Note that this type of animation requires some camera that provides real-time video stream (i.e., web camera). 
Firstly, you will see the window with your face - you need to make a neutral facial expression and press 'Enter'. 

![](https://github.com/NikitaBogoslovskiy/AvatarAnimation/blob/dev/other_data/readme_files/neutral_face.png?raw=true)

Then you will observe two windows: one with your face, another with a character which is being animated in real-time. You can control character animation with your facial expressions. 

![](https://github.com/NikitaBogoslovskiy/AvatarAnimation/blob/dev/other_data/readme_files/online_video_animation.png?raw=true)

#### Offline mode
```python
video_animation_offline(video_path="path/to/video", photo_path="path/to/photo")
```
This type of animation works with pre-recorded video so you need to specify the path to the video with your facial expressions. Also the program requires a path to the photo with your neutral face. If you do not specify the path to the photo, the program will consider that you have neutral expression at the beginning of the video.

The result of animation will be saved to the same directory with name "<origin_video_name>_output.mp4" and look like:

![](https://github.com/NikitaBogoslovskiy/AvatarAnimation/blob/dev/other_data/readme_files/offline_video_animation.gif)

### Audio animation
```python
audio_animation_offline(audio_path="path/to/audio")
```
Audio animation takes the audio wav-file with human speech and generate animation with a character whose lips are synchronized with recognized speech. Note that we developed this part of the project only for Russian speech. In particular, we used already trained for Russian language speech-to-text model wav2vec 2.0.  

The result of animation will be saved to the same directory with name "<origin_audio_name>.mp4" and look like:

![](https://github.com/NikitaBogoslovskiy/AvatarAnimation/blob/dev/other_data/readme_files/audio_animation.gif)

### Overall animation
```python
overall_animation_offline(video_path=f"path/to/video", audio_support_level=0.8)
```
This type takes pre-recorded video clip with facial expressions and speech and then creates a complex animation:
- upper facial parts (forehead, eyebrows, eyes, nose) are animated using only facial expressions from video stream
- lips are animated in a hybrid way - during speechless video fragments lips are animated due to video stream, during fragments with speech lips are animated based both on video stream and speech from audio track (audio_support_level defines the influence of audio track during fragments with speech and varies from 0.0 to 1.0, where 0.0 - no influence, 1.0 - full audio animation).

The result of animation will be saved to the same directory with name "<origin_video_name>_output.mp4".

## Acknowledgements

For animation we used parameterized model of the head [FLAME](https://flame.is.tue.mpg.de/). In particular, we worked with its PyTorch [implementation](https://github.com/soubhiksanyal/FLAME_PyTorch).
