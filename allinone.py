from PIL import Image
from rembg.bg import remove
import numpy as np
import io
import cv2
import os

def preprocess_image(path):
  path_new = path
  img = Image.open(path)
  extension = img.format

  need_conversion = False

  if (extension != 'PNG'):
      need_conversion = True
      img.save("png.png")
      path_new= 'png.png'

  im_a = np.fromfile(path_new)
  im_no_bg = remove(im_a)
  im_seg = Image.open(io.BytesIO(im_no_bg)).convert("RGBA")
  im_seg.save("seg.png")

  x,y,w,h = 0,0,0,0
  im_bgr = cv2.imread("seg.png")
  im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
  im_rgb_copy = im_rgb.copy()

  lower = np.array([30,30,30])
  higher = np.array([250,250,250])

  mask = cv2.inRange(im_rgb, lower, higher)

  cont,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  c = max(cont, key = cv2.contourArea)

  x,y,w,h = cv2.boundingRect(c)

  if (x > 50 and y > 50):
    x = x - 50
    y = y - 50
    w = w + 100
    h = h + 100 

  cv2.rectangle(im_rgb, (x,y), (x+w,y+h), (0,255,0), 5)
  im_cr = im_rgb_copy[y:y+h, x:x+w]
  Image.fromarray(im_cr).save('final.jpg')
  path_new = 'final.jpg'
  if need_conversion == True:
    os.remove('png.png')
  os.remove('seg.png')
  
  return path_new
