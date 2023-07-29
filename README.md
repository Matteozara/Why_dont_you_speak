# Why don’t you speak? (WDyS)
This repo contains the video demo for the paper "Why don’t you speak?: a smartphone application to engage
museum visitors through Deepfake of statues".
<br>
The paper presents a new application to change the way of experience the museums.
To understand better the functionalities, here (<a href="#project-videos">Project Videos</a>) there are three video, each one explaining a different feature of the app.
<br>
<!-- To have a full overview, you can donwload the video [video_demo.mp4](https://drive.google.com/file/d/1eKe5PnxXyIel1fs0ag8DvSHfpzSxyI3T/view?usp=sharing). -->
<br>
If you just want to see the final deep fakes you can go to <a href="#results">Results</a>, or, to have more examples, search inside the <i>"Deep fake Results" </i> folder.
<br>
<br>
<!-- TABLE OF CONTENTS -->
  ### Table of contents
  <ol>
    <li>
      <a href="#project-videos">Project Videos</a>
      <ul>
        <li><a href="#create-the-deep-fake">Create The Deep Fake</a></li>
        <li><a href="#know-your-statue">Know Your Statue</a></li>
        <li><a href="#add-your-audio">Add Your Audio</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#server-application">Server Application</a></li>
        <li><a href="#android-application">Android application</a></li>
      </ul>
    </li>
    <li><a href="#results">Results</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contacts">Contacts</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
  
<br>
<br>

## Project Videos
#### Create The Deep Fake
This is the first feature and it allows to create a deep fake based on a video live recorded:

https://github.com/Matteozara/Why_dont_you_speak/assets/74371691/34c6200b-eddf-4e0d-8f12-9be720bcdc40


#### Know Your Statue

This feature allows people to know the statue recorded, because it shows its history:

https://github.com/Matteozara/Why_dont_you_speak/assets/74371691/737536a4-728c-4f37-9dc4-eba1372a75a8


#### Add Your Audio

This is an amusement functionality that allows to record an audio and create a deep fake based on the audio just recorded;

https://github.com/Matteozara/Why_dont_you_speak/assets/74371691/b779d4db-b512-4613-bab0-93bc02d51281



<br>
<br>

## Getting Started
<!--https://github.com/Matteozara/Why_dont_you_speak.git-->

First of all clone the repo:
```sh
  git clone "url repo"
  cd Why_dont_you_speak
  ```
NB: to run the project you need that both server_application and Android_application are connected to the same network.
<br>
<br>

### Server Application

To run the Server Application, first of all, set up an environment with GPU.
<br>
<br>
After, donwload the weights of the models [link here](https://drive.google.com/drive/folders/1EwbSPdOrXYlIqTS0SufuodawTS7eR-1P?usp=drive_link), put the EfficientNet weights (both), iside the <i>server_application/EfficientNet</i> folder, the Yolo8 weights inside the <i>server_application/Yolo8</i> folder, and the GAN weights inside <i>server_application/model</i> folder.
<br>
<br>
Once done, install the required packages:
```sh
  pip install torch
  pip install opencv-python
  pip install torchvision
  pip install facenet-pytorch
  pip install ffmpeg-python  or   sudo apt install ffmpeg
  python -m pip install librosa  
  ```
<br>
<br>
After installed all the packages, you should go inside Server Application and run the server:
```sh
  cd server_application
  python server_flaskGPU.py
  ```
<br>
<br>
<b>PS</b>: Inside the <i>server_flaskGPU.py</i> file, you have to change the server and the port based on your network (last line of code):
<br>
app.run(host="<i>ip_server</i>", port=<i>port_number</i>, threaded=True)
<br>
<br>

### Android Application
Open the <I>WDyS</i> folder with Andoird Studio. 
<br>
<br>
Change the <i>server</i> String variable in <i>ExplanationActivity.java, MainActivity.java, AudioActivity.java</i> and <i>VideoAudioActivity.java</i>, with your server address, the one you wrote inside <i>server_flaskGPU.py</i> (you have to be sure that both Server and Andorid application run on the same network).
<br>
<br>
Run the app on your Android smartphone and try it.

<br>
<br>

## Results
The result is shown directly in the Andoird application (where there is also the possibility to save it in the gallery), but is also saved inside the <i>server_application/final_results</i> on server side.
Here there are two examples of deep fake generated, if you want to see more look inside the <i>"Deep fake Results" </i> folder.

https://github.com/Matteozara/Why_dont_you_speak/assets/74371691/f24b4a87-366f-4a1b-a444-ddda50077f74

https://github.com/Matteozara/Why_dont_you_speak/assets/74371691/65aefe89-3196-44b5-8c9b-cd8640f80f91

  
<br>
<br>

## License
This repository can only be used for personal/research/non-commercial purposes. However, for commercial requests, please contact us directly <a href="#contacts">contacts</a>
  
<br>
<br>

## Contacts
TBA
<!-- Matteo Zaramella - zaramella.2025806@studenti.uniroma1.it or matteozara98@gmail.com

Irene Amerini - amerini@diag.uniroma1.it

Paolo Russo - prusso@diag.uniroma1.it-->
  
<br>
<br>

## Acknowledgments

The code for the Deep fake generation (Generative Adversarial Network) has been taken from the [Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip). We thank the authors for releasing their code and models.
