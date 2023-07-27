import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
from PIL import Image

#check if there is a face in the first frame (the video is static, so if is good the first frames, hopefully is good all the video)
#(return 'False' if there isn't any face or if there is more than one face)

def check_faces(image, yolo):
  #image = image.convert('RGB')
  #image_array = np.array(image, dtype=np.float32)
    # cv2 image color conversion
  image_array = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  try:
    boxes, _, _, _ = yolo.detect(image)
  except:
    return False
  
  if len(boxes) > 0:
    return True
  
  return False

def check_all_frames_statue(video, yolo, model_back, model, device):
  results_HS = [0,0]
  indices_to_labels = ['arringatore', 'atena', 'atena_armata', 'demostene', 'dioniso', 'era_barberini', 'ercole', 'kouros_da_tenea', 'minerva_tritonia', 'poseidone', 'zeus', 'altro']
  result = [0,0,0,0,0,0,0,0,0,0,0,0] #12
  #t = transforms.ToPILImage()

  #loop
  for i in video:
    #face detection
    if not check_faces(i, yolo):
      print("Error face not detected")
      return -1
    
    #classify human statue
    results_HS[classify_statue_human_single(i, model_back, device)] += 1

  if results_HS[0] >= (len(video)/2): #human
    print("Error human present")
    return -1
  
  for i in video:
    result[classify_statue_single(i, model, device)] += 1

  index = 0
  for i in range(0,len(result)):
    if  result[i] > result[index]:
      index = i
  print(result)
  return indices_to_labels[index]



def check_all_frames_human(video, yolo):
  #t = transforms.ToPILImage()

  #loop
  for i in video:
    #face detection
    if not check_faces(i, yolo):
      print("Error face not detected")
      return -1
    
  return 0


def check_single_frame_statue(frame, yolo, model_back, model, device):
  indices_to_labels = ['arringatore', 'atena', 'atena_armata', 'demostene', 'dioniso', 'era_barberini', 'ercole', 'kouros_da_tenea', 'minerva_tritonia', 'poseidone', 'zeus', 'altro']
  #t = transforms.ToPILImage()

  #face detection
  if not check_faces(frame, yolo):
    print("Error face not detected")
    return -1
  
  #classify human statue
  if classify_statue_human_single(frame, model_back, device) == 0:  #human
    print("Error human present")
    return -1
  
  return indices_to_labels[classify_statue_single(frame, model, device)]



def check_single_frame_human(frame, yolo):
  #t = transforms.ToPILImage()
  
  if not check_faces(frame, yolo):
    print("Error face not detected")
    return -1
    
  return 0



def classify_statue_human_single(frame, model_back, device):
  #indices_to_labels = ['human', 'statue']
  device = torch.device(device)
  t = transforms.ToPILImage()
  cut = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
  image = t(frame)
  img = cut(image)
  batch_t = torch.unsqueeze(img, 0)
  batch_t = batch_t.to(device)
  out = model_back(batch_t)
  _, index = torch.max(out, 1)
  
  index = index.item()
  return index


def classify_statue_single(img, model, device):
  device = torch.device(device)
  t = transforms.ToPILImage()
  cut = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
  image = t(img)
  img = cut(image)
  batch_t = torch.unsqueeze(img, 0)
  batch_t = batch_t.to(device)
  out = model(batch_t)
  _, index = torch.max(out, 1)
  percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
  index = index.item()
  if percentage[index].item() < 50: #set thrashold to 'Altro'
    return 11
  else:
    return index
  