import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import scipy, cv2, os, sys
from os import listdir, path
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob

import torch
import platform

import librosa
import librosa.filters
import numpy as np
# import tensorflow as tf
from scipy import signal
from scipy.io import wavfile


def melspectrogram(wav):
    ##preemphasis => apply a digital filter (if hp.preemphasize = True)
    ##_stft => return a Fourier Transformation
    wav = signal.lfilter([1, -0.97], [1], wav)
    D = librosa.stft(y=wav, n_fft=800, hop_length=200, win_length=800)


    mel_bais = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    mel = np.dot(mel_bais, np.abs(D))
    min_level = np.exp(-100 / 20 * np.log(10))
    min = 20 * np.log10(np.maximum(min_level, mel))
    S = min - 20
    
    return np.clip((2 * 4.) * ((S + 100) / (100)) - 4., -4., 4.)



#datagen
def datagen(frames, mels, yolo):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    using_video_format = True
    face_det_results = []
    cropped = []
	##face_detect() return an array of immages cropped and coordinates with only faces (obtained with face detection)
    if using_video_format:
		#face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        for i in range(0, len(frames)):
            image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            try:
                result, _, _, _ = yolo.detect(image)
            
                if result != []:
                    face_det_results.append((round(result[0][1]), round(result[0][1] + result[0][3]), round(result[0][0]), round(result[0][0] + result[0][2])))
                    a = frames[i][round(result[0][1]):round(result[0][1]+result[0][3]), round(result[0][0]):round(result[0][0]+result[0][2])]
                    cropped.append(a)
            except:
                if len(face_det_results) > 0:
                    face_det_results.append(face_det_results[-1])
                    cropped.append(cropped[-1])
    else:
        #face_det_results = face_detect([frames[0]])
        image = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        try:
            result, _, _, _ = yolo.detect(image)
            
            if result != []:
                face_det_results.append((round(result[0][1]), round(result[0][1] + result[0][3]), round(result[0][0]), round(result[0][0] + result[0][2])))
                a = frames[i][round(result[0][1]):round(result[0][1]+result[0][3]), round(result[0][0]):round(result[0][0]+result[0][2])]
                cropped.append(a)
        except:
            raise Exception("In the image is not detected any face")

    #hoping the faces missed are only a few (adjust dimension)
    if len(frames) > len(face_det_results):
        for i in range(0, (len(frames) - len(face_det_results))):
            face_det_results.append(face_det_results[-1])
            cropped.append(cropped[-1])

	##process and adapted frames with audio file
    for i, m in enumerate(mels):
        idx = i%len(frames) #idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy() ##original frame
        face = cropped[idx].copy()
        coords = face_det_results[idx]
        #face, coords = face_det_results[idx].copy() ##cropped frame + coords

        face = cv2.resize(face, (96, 96)) ##args.img_size = 96 (by default)  [why?]
			
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
		
		##reshape length to be conformed to batch_size(?)
        if len(img_batch) >= 128:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch) ##np.asarray => transform input arg in to array

            img_masked = img_batch.copy()
            img_masked[:, 96//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch ##returns the generator for these lists
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, 96//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch ##returns the generator for these lists





def start_generating(model, yolo, full_frames, path_wav, iteration,device):
    outputs = []
    fps = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    mel_step_size = 16
    #audio_path = ["audio1.wav", "audio2.wav", "audio3.wav"]
    result_path = 'final_results/final_result{}.mp4'.format(iteration)
    result_temp_path =  'temp_results/result_temp{}.avi'.format(iteration)
        
    ##Load audio
    #path_wav = "audio/" + audio_path[id_audio]
    wav = librosa.core.load(path_wav, sr=16000)[0]
    mel = melspectrogram(wav)

    ##np.isnan => return true if inside there is not a number
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    
    ##process audio
    mel_chunks = []
    mel_idx_multiplier = 80./fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    ##adapt video to the length of audio
    full_frames = full_frames[:len(mel_chunks)]

    batch_size = 128

    ##########################
    #start.record()
    ##########################

    #print("start datagen")
    gen = datagen(full_frames.copy(), mel_chunks, yolo) ##returns the generator for lists: img_batch(faces), mel_batch(audio), frame_batch(original frames), coords_batch(coords of faces)
    #print("end datagen")

    ##np.ceil(a) => return all the elemnt of 'a' list, rounded on top (es 0.1 => 1)
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):

        ##only first iteration
        if i == 0:
            #model = load_model(args.checkpoint_path)
            ##set height and weight
            frame_h, frame_w = full_frames[0].shape[:-1]
            #save video in 'temp/result.avi'
            out = cv2.VideoWriter(result_temp_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        ##[why transpose?] (maybe for the model)
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
    
        #torch.save(img_batch, './img.pt')
        #torch.save(mel_batch, './mel.pt')

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        ##[why contro-transpose?] (maybe decode the output of the model)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c ##unzip coords
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1)) ##convert tensor p in np.uint8 and then resize

            f[y1:y2, x1:x2] = p ##overwrite pred on the original frame
            ###########f is the new frame (modified)
            outputs.append(f)
            out.write(f) ##save all on the temp file ('temp/result.avi')

    out.release()

    #print("end, right now save")

    #save the temp file ('temp/result.avi') on the output path (default: 'results/result_voice.mp4')
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(path_wav, result_temp_path, result_path)
    #command = 'ffmpeg -i {} -i {} -c:v copy -c:a aac {}'.format(result_temp_path, path_wav, result_path)
    #subprocess.call(command, shell=platform.system())
    subprocess.call(command, shell=platform.system() != 'Windows')

    print("END!!!")
    return outputs, result_path

    ##############################################
    #end.record()
    #torch.cuda.synchronize()
    #print('Time: ', start.elapsed_time(end))
    ###############################################





