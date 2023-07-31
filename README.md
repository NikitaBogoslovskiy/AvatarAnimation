# Avatar Animation

This is the repository for the paper "Using Neural Networks to Mitigate Articulation Challenges in Speech-Driven Lip-Sync Models" by Nikita Bogoslovskiy and Michael Yurushkin.

## Research paper description

Within the framework of this research, the problem of preparing datasets for lip-sync models that allow performing animation based on audio recordings with speech is considered. In particular, an approach to solving this problem is being studied, which consists in recording a series of videos, on each of which a person utters one sentence, and building a 3D model whose lips are synchronized with the lips of the speaker. As part of this approach, a video-based animation model is used, working on the basis of neural networks and transferring human facial expressions to a 3D character. This approach is demanding for a person, because throughout all the videos their articulation should be at a consistently good level. To improve articulation when transferring facial expressions, we are upgrading the video-based animation model, namely:
* Changing the architecture of the neural network
* Introducing a new loss function that allows the neural network to focus on the labial part during training
* We introduce a heuristic that provides more expressive articulation at the stage of model execution

## Related materials

The implementation of the upgraded video-based animation model, as well as the lip-sync model performing audio-based animation, is in this repository. Generated datasets, comparative demo videos and videos with a speaker can be found in [Google Drive](https://drive.google.com/drive/folders/16cVTVJXoDFbNaz3EhOAknhLMbSDOQnyN?usp=sharing).
