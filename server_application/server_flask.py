from flask import Flask, send_file, request
import base64
import numpy as np
from find_statue import check_single_frame_human, check_single_frame_statue, check_all_frames_human, check_all_frames_statue
from deep_fake import start_generating
import torch
import cv2
from yolov8face import YOLOv8_face
import torchvision.transforms as transforms
import os



app = Flask(__name__)

#kind of Main
#load model
model = torch.jit.load("model/gan_gpu_jit.pt") #or "gan_cpu_PC_jit.pt" for cpu
model.eval()
#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#face detector
yolo = YOLOv8_face("Yolo8/yolov8n-face.onnx", conf_thres=0.45, iou_thres=0.5)

#statue vs human classifier
model_back = torch.load('EfficientNet/EfficientNet_back.pt', map_location=device)
model_back.to(device)
model_back.eval()

#type of statue classifier
model_statue = torch.load('EfficientNet/EfficientNet_video.pt', map_location=device)
model_statue.to(device)
model_statue.eval()


## STATUE ##
def start_process_statue(path, index, tipo=0):
    #split video in frames
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    label = ""

    video = []
    c = 0
    while 1:
        still_reading, frame = vidcap.read()
        if not still_reading:
            vidcap.release()
            break
                        
        x1 = 0
        y1 = 0
        x2 = frame.shape[1]
        y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        video.append(frame)


    ## FIND STATUE ##
    if tipo == 0:
        label = check_single_frame_statue(video[0], yolo, model_back, model_statue, device)
        print("run single frame done")
        if label == -1:
            return "Error"
    else:
        label = check_all_frames_statue(video, yolo, model_back, model_statue, device)
        print("run all frames done")
        if label == -1:
            return "Error"
    #set audio path (based on label found)
    audio = "Audio_wav/" + label + ".wav"
    print(audio)

    ## DEEPFAKE ##
    result, path_result_file = start_generating(model, yolo, video, audio, index)

    print("Path: ",path_result_file)
    return path_result_file



## INFO ##
def start_process_info(path, index, tipo=0):
    label = ""
    #split video in frames
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    video = []
    c = 0
    while 1:
        still_reading, frame = vidcap.read()
        if not still_reading:
            vidcap.release()
            break
                        
        x1 = 0
        y1 = 0
        x2 = frame.shape[1]
        y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        video.append(frame)

    ## FIND STATUE ##
    if tipo == 0:
        label = check_single_frame_statue(video[0], yolo, model_back, model_statue, device)
        print("run single frame done")
        if label == -1:
            return "Error"
    else:
        label = check_all_frames_statue(video, yolo, model_back, model_statue, device)
        print("run all frames done")
        if label == -1:
            return "Error"
    
    return label


## PEOPLE ##
def start_process_people(path, index, tipo=0):
    #split video in frames
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    video = []
    c = 0
    while 1:
        still_reading, frame = vidcap.read()
        if not still_reading:
            vidcap.release()
            break
                        
        x1 = 0
        y1 = 0
        x2 = frame.shape[1]
        y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        video.append(frame)

    if tipo == 0:
        label = check_single_frame_human(video[0], yolo)
        print("run single frame done")
        if label == -1:
            return "Error"
    else:
        label = check_all_frames_human(video, yolo)
        print("run all frames done")
        if label == -1:
            return "Error"
    if label == -1:
        return "Error"
    
    #set audio path (based on label found)
    audio = "Audio_wav/altro.wav"
    print(audio)
    result, path_result_file = start_generating(model, yolo, video, audio, index)

    print("Path: ",path_result_file)
    return path_result_file


## PEOPLE AND STATUE CUSTOM##
def start_process_custom(path, index, tipo=0):
    #split video in frames
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    video = []
    c = 0
    while 1:
        still_reading, frame = vidcap.read()
        if not still_reading:
            vidcap.release()
            break
                        
        x1 = 0
        y1 = 0
        x2 = frame.shape[1]
        y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        video.append(frame)

    if tipo == 0:
        label = check_single_frame_human(video[0], yolo)
        print("run sinle frame done")
        if label == -1:
            return "Error"
    else:
        label = check_all_frames_human(video, yolo)
        print("run all frames done")
        if label == -1:
            return "Error"
    if label == -1:
        return "Error"

    audio = "Custom_audio/audio.wav"
    
    result, path_result_file = start_generating(model, yolo, video, audio, index)

    print("Path: ",path_result_file)
    return path_result_file


@app.route('/video_display', methods=['GET'])
def get_video():
    if os.path.isfile('final_results/final_result0.mp4'):
        return send_file('final_results/final_result0.mp4', mimetype='video/mp4')
    else:
        return "Error"


@app.route('/video_give_statue', methods=['POST'])
def json_handler_statue():
    # Get the JSON file from the request
    json_file = request.get_json()
    video_data = json_file['video']
    index = json_file['index']
    tipo = json_file['type']

    video_data = base64.b64decode(video_data)

    path = "save_video/new" + str(index) + ".mp4"
    #save video
    with open(path, "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(video_data)

    return start_process_statue(path, index, tipo) #add send file back


@app.route('/video_give_people', methods=['POST'])
def json_handler_people():
    # Get the JSON file from the request
    json_file = request.get_json()
    video_data = json_file['video']
    index = json_file['index']
    tipo = json_file['type']

    video_data = base64.b64decode(video_data)

    path = "save_video/new" + str(index) + ".mp4"
    #save video
    with open(path, "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(video_data)

    return start_process_people(path, index, tipo)


@app.route('/check_statue', methods=['POST'])
def check_statue():
    # Get the JSON file from the request
    json_file = request.get_json()
    video_data = json_file['video']
    index = json_file['index']
    tipo = json_file['type']

    video_data = base64.b64decode(video_data)

    path = "save_video/new" + str(index) + ".mp4"
    #save video
    with open(path, "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(video_data)

    return start_process_info(path, index, tipo)


#save and convert custom audio
@app.route('/send_audio', methods=['POST'])
def send_audio():
    # Get the JSON file from the request
    json_file = request.get_json()
    audio_data = json_file['audio']

    audio_data = base64.b64decode(audio_data)

    #save audio
    with open("Custom_audio/audio.3gp", "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(audio_data)

    try:
        os.system('ffmpeg -y -i Custom_audio/audio.3gp Custom_audio/audio.wav')
        return "done"
    except:
        print("Error on saving and converting the audio")
        return "Error"
    

@app.route('/process_custom', methods=['POST'])
def process_custom():
    # Get the JSON file from the request
    json_file = request.get_json()
    video_data = json_file['video']
    index = json_file['index']
    tipo = json_file['type']

    video_data = base64.b64decode(video_data)

    path = "save_video/new" + str(index) + ".mp4"
    #save video
    with open(path, "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(video_data)

    return start_process_custom(path, index, tipo)



if __name__ == '__main__':
    app.run(host="172.20.10.5", port=3535, threaded=True) # processes=3 => up to 3 processes    server = 172.20.10.5

