import os
import shutil
import numpy as np
import cv2

import argparse
import re

from pathlib import Path
from typing import Optional
from skimage import io

from saliency_models.DIS import Saliency_ISNET_Node
from saliency_models.U2Net import Saliency_U2Net_Node


VALID_IMAGE_EXTENSION = (
    'rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff', 'rast',
    'xbm', 'jpeg', 'jpg', 'bmp', 'png', 'webp', 'ext')

def parse_args():
    parser = argparse.ArgumentParser(description='Saliency Object Detection for segmentation map')
    parser.add_argument('--characters_path', default='output/characters', help='source directory')
    parser.add_argument('--save_path', default='./output/mask', help='saving path generated scene images')
    parser.add_argument('--isnet-model-name', type=str, default='isnet-general-use', help='checkpoint name')
    args = parser.parse_args()
    return args

class SaliencyNode:
    def __init__(self,
                model_path: str = os.path.join('saliency_models', 'DIS'), # model path
                device: str = "cuda",
                model_name: str = "isnet"):
        self.device = device
        self.models = {}
        # print("before init")
        self.init(model_path, model_name)
        # print("after init") # 작동 확인완
        
    def init(self, model_path: str='saliency_models', model_name: str='u2net'):
        if model_name.startswith("u2net"):
            self.models[model_name] = Saliency_U2Net_Node(model_path, device=self.device, model_name=model_name)
            # print("u2net")
        elif model_name.startswith("isnet"):
            self.models[model_name] = Saliency_ISNET_Node(model_path, device=self.device, model_name=model_name)
            # print("isnet")
        else:
            print(f"[SaliencyNode] ERROR Unrecognized model name. Choose from [u2net|isnet]. You provided {model_name}")

    def __call__(self, img, model_name='isnet-generatl-use'):
        if model_name in self.models.keys():
            # print("model __call__ keys")
            return self.models[model_name](img)
        else:
            # print("model __call__ else")
            self.init(model_name)
            return self.models[model_name](img)

def run_image(filename, model,
              dest: Optional[str] = None, # target directory
              ext: str = 'png', # output file extension
              isnet_model_name: str = 'isnet-general-use', # model name
    ):
    # Check if it is a file
    if os.path.isfile(filename):# and "result" not in filename:
        img_name = Path(filename).stem

        # Segment foreground
        img = io.imread(filename)

        if img.ndim == 2: # gray-scale image
            img = np.stack([img]*3, axis=-1)

        res = model(img, model_name=isnet_model_name)

        # # #########################
        # from PIL import Image
        # # print(filename)
        # # image = Image.open(filename)
        # mask = Image.fromarray(res[0].astype(np.uint8))
        # # alpha = Image.new("L", mask.size, 0)
        
        # # im = Image.composite(mask, image, alpha)
        # # im.save(os.path.join(dest, f"{img_name}_foreground.{ext}"))
        
        # import cv2
        
        # image = cv2.imread(filename)
        # masked = cv2.bitwise_not(image, image, mask=res[0].astype(np.uint8))
        # cv2.imwrite(os.path.join(dest, f"{img_name}_foreground.{ext}"), masked)
        
        
        # Save mask
        io.imsave(os.path.join(dest, f"{img_name}_mask.{ext}"), res[0].astype(np.uint8))
        print("io save")

        saliency_mask = res[0].astype(np.float32) / 255. # turn to 0-1 values
        segmented_fg = np.expand_dims(saliency_mask, 2) * img.astype(np.float32) # mask out background
        # Save segmented foreground
        io.imsave(os.path.join(dest, f"{img_name}_foreground.{ext}"),
                    segmented_fg.astype(np.uint8))
        print("io imsave")

def crop_size(filename, dest, ext: str = 'png', extra=None):
    img_name = Path(filename).stem
    
    input_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # height = input_image.shape[0]
    # width = input_image.shape[1]
    
    if len(input_image.shape) == 2:
        gray_input_image = input_image.copy()
    else:
        gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) 
        
    # To find upper threshold, we need to apply Otsu's thresholding
    upper_threshold, thresh_input_image = cv2.threshold(
        gray_input_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    lower_threshold = 0.5 * upper_threshold
    
    canny = cv2.Canny(input_image, lower_threshold, upper_threshold)
    # Finding the non-zero points of canny
    pts = np.argwhere(canny > 0)
    
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)

    # Crop ROI from the givn image
    output_image = input_image[y1:y2, x1:x2]
    cv2.imwrite(filename, output_image)
    # cv2.imwrite(os.path.join(dest, f"{img_name}.{ext}"), output_image)
    
    # if extra is not None:
    extra_image = cv2.imread(extra, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(extra, extra_image[y1:y2, x1:x2])
    # return x1, y1, x2, y2


if __name__ == '__main__':

    args = parse_args()
    
    model = SaliencyNode(model_path=os.path.join('saliency_models', 'DIS'), model_name=args.isnet_model_name)
    Path(args.save_path).mkdir(exist_ok=True, parents=True)
    
    print("Generation segmentation map using saliancy map")
    
    characters = os.listdir(args.characters_path)
    # characters = [os.path.splitext(file)[0] for file in characters] # extract only file names (123.jpg -> 123)
    # extensions = [os.path.splitext(file)[1] for file in characters] # 
    
    for character in characters:
        path = args.characters_path +'/'+character
        run_image(path, model=model, dest=args.save_path)
        
        character_path = args.save_path+"/"+re.sub('.png','',character)
        print(character_path)
        
        # crop_size(character_path+"_mask.png", args.save_path, extra=character_path+"_foreground.png")