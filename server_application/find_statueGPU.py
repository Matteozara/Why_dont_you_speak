import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN

#check if there is a face in the first frame (the video is static, so if is good the first frames, hopefully is good all the video)
#(return 'False' if there isn't any face or if there is more than one face)

def check_faces(image, mtcnn):
  print("check faces")
  image = image.convert('RGB')
  image_array = np.array(image, dtype=np.float32)
    # cv2 image color conversion
  image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
  bounding_boxes, conf = mtcnn.detect(image_array)
  if conf[0] == None:
    return False
  if len(conf) == 1 and conf[0] > 0.59:
    return True
  
  return False



#classifier statue/human
  
def classify_statue_human(video_frames, model_back, device):
  indices_to_labels = ['human', 'statue']
  result = [0,0] #11
  device = torch.device(device)
  t = transforms.ToPILImage()
  cut = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
  """PATH = 'EfficientNet/EfficientNet_back.pt'
  model_back = torch.load(PATH, map_location=device)
  model_back.to(device)
  model_back.eval()"""
  for img in video_frames:
    image = t(img)
    img = cut(image)
    #img = torch.rot90(img, 1, [2, 1]) #rotate image of 90 degrees to make it right
    batch_t = torch.unsqueeze(img, 0)
    batch_t = batch_t.to(device)
    out = model_back(batch_t)
    _, index = torch.max(out, 1)
    #percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    index = index.item()
    #print(percentage[index].item())
    result[index] += 1

  if result[0] >= 7:
    return indices_to_labels[0]
  else:
    return indices_to_labels[1]


#classifier function to classify wich statue is present on video
  
def classify_frames(video_frames, model, device):
  indices_to_labels = ['arringatore', 'atena', 'atena_armata', 'demostene', 'dioniso', 'era_barberini', 'ercole', 'kouros_da_tenea', 'minerva_tritonia', 'poseidone', 'zeus', 'altro']
  result = [0,0,0,0,0,0,0,0,0,0,0,0] #12
  device = torch.device(device)
  t = transforms.ToPILImage()
  cut = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
  """PATH2 = 'EfficientNet/EfficientNet_video.pt'
  model = torch.load(PATH2, map_location=device)
  model.to(device)
  model.eval()"""
  for img in video_frames:
    image = t(img)
    img = cut(image)
    #img = torch.rot90(img, 1, [2, 1]) #rotate image of 90 degrees to make it right
    batch_t = torch.unsqueeze(img, 0)
    batch_t = batch_t.to(device)
    out = model(batch_t)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    index = index.item()
    #print('Percentuale: ', percentage[index].item(), ' Classe: ', indices_to_labels[index])
    if percentage[index].item() < 50: #set thrashold to 'Altro'
      result[11] += 1
    else:
      result[index] += 1

  
  index = 0
  for i in range(0,len(result)):
    if  result[i] > result[index]:
      index = i
  print(result)
  return indices_to_labels[index]


#function to select random frames

def random_numbers(video_frames):
  number = []
  for i in range(0, 13):
    number.append(random.randrange(0, len(video_frames)-2))
  return number

def select_random_frames(video_framse):
  number = random_numbers(video_framse)

  lista_video = []
  for i in number:
    lista_video.append(video_framse[i])
  return lista_video


def run(video, image, device, mtcnn, model_back, model_statue):
    print("run1")

    lista_video = select_random_frames(video) #random frames

    #face detector
    face = check_faces(image, mtcnn)
    #if the def return false, the algorithm will stop, because can't produces an output in any cases
    if not face:
        print('On the video there isn\'t any face or there are more than one, in both cases, the algorithm can\'t work. Please, do another video with a single face')
        return -1


    #lista_video = select_random_frames(video) #random numbers
    classe = classify_statue_human(lista_video, model_back, device)
    print(classe)

    #if the 'classify_statue_human' return human, the algorithm will stop.
    if classe == 'human':
        print('The video represent a human, try again with a statue video (from Sapienza\'s Gipsoteca)')
        return -1

    label = classify_frames(lista_video, model_statue, device)
    print(label)

    return label


def run_people(image, mtcnn):
    print("run1")

    #face detector
    face = check_faces(image, mtcnn)
    #if the def return false, the algorithm will stop, because can't produces an output in any cases
    if not face:
        print('On the video there isn\'t any face or there are more than one, in both cases, the algorithm can\'t work. Please, do another video with a single face')
        return -1

    return 0
