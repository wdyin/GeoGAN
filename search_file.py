from PIL import Image,ImageChops
import sys,os
import json
from torchvision import transforms
import math, operator
from functools import reduce
def equal(im1, im2):
  bbox = ImageChops.difference(im1, im2).getbbox()
  print(bbox)
  return bbox is None



def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"

    h = ImageChops.difference(im1, im2).histogram()

    # calculate rms
    return math.sqrt(reduce(operator.add, map(lambda h, i: h*(i**2), h, range(256))) / (float(im1.size[0]) * im1.size[1]))

ref_files = os.listdir(sys.argv[1])
ref_dict = {}
trans = transforms.Compose([transforms.CenterCrop(178),
                           transforms.Resize(256)])

for filename in ref_files:
    img = Image.open(os.path.join(sys.argv[1],filename))
    img = trans(img)
    ref_dict[filename] = img

result_dict = {}
for num in range(0,30):
    img_name = 'epoch000_{}_By_0.jpg'.format(num)
    im = Image.open(os.path.join(sys.argv[2],img_name))
    for another_im in ref_dict:
        diff= rmsdiff(im,ref_dict[another_im])
        if diff < 5:
            print(diff)
            result_dict[num] = another_im
            print(num)

with open(sys.argv[3],'w') as outfile:
    json.dump(result_dict,outfile)

