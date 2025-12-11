from openai import OpenAI
import openai
import inflect
import argparse
import re
import cv2
import PIL
import plotext as plotext
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from tqdm import tqdm
import imghdr
from pathlib import Path
from PIL import Image
from torchvision.ops import box_iou
from termcolor import colored
from collections import defaultdict
from rich.console import Console
from rich.markdown import Markdown

import math
import numpy as np
import random
import os
import pandas as pd
import os.path as osp
import json
import time
import torch

from typing import Optional

from template import ratio_template, box_template, ratio_prompt, box_prompt


OPENAI_API_BASE = "https://api.openai.com/v1/" # NOTE: [User specified]
OPENAI_API_KEY = "sk-"
OPENAI_MODEL ='gpt-4' 

def show_report(report_list: list=['> Report']):
    console = Console()

    markdown = Markdown('\n'.join(report_list))
    console.print(markdown)
    
def get_objects(d):
    # Use the 'max' and 'min' functions to find the key corresponding to the maximum and minimum values in the dictionary.
    # The 'key' argument specifies that the key should be determined based on the values using 'd.get'.
    return max(d, key=d.get), min(d, key=d.get)

def parse_args():
    parser = argparse.ArgumentParser(description='Samples Creator')
    parser.add_argument('--height', default=1024, type=int, help='image height')
    parser.add_argument('--width', default=1024, type=int, help='image width')
    
    parser.add_argument('--scene_path', default='output/story/scene.txt', help="scene path prompt")
    parser.add_argument('--input_characters', type=str, default="output/story/characters.json", help="character names and features")
    parser.add_argument('--save_path', default='output/story', type=str, help='json save path')
    
    parser.add_argument('-obj_ratio', '--with_obj_ratio', action='store_true', help='Get object ratio or not')
    
    args = parser.parse_args()
    return args

class Text2Box(object):
    def __init__(
        self,
        prompt, 
        charcters,
        img_height: int = 1024,
        img_width: int = 1024,
        bboxes_examples: Optional[list] =None,
    ):
        # self.init_api_settings()
        
        self.bboxes_examples = bboxes_examples
        
        self.img_height = img_height
        self.img_width = img_width
        
        self.prompt = prompt
        self.charcters = charcters
    
    # def init_api_settings(self):
    #     openai.api_base = OPENAI_API_BASE
    #     openai.api_key = OPENAI_API_KEY
    
    def obj_size_reactions(self):
        template = ratio_template + ratio_prompt
        prompt = template.format(prompt=self.charcters)
        client = OpenAI(
            api_key = OPENAI_API_KEY, 
        )
        
        response = client.chat.completions.create(
                model = OPENAI_MODEL,
                messages =[
                    {"role": "user", "content" : f"""{prompt}"""}
                ],  
        )
        return response.choices[0].message.content
    
    def get_layout(self):
        if bool(self.bboxes_examples):
            template = box_template + ''.join(self.bboxes_examples) + box_prompt
        else:
            template = box_template + box_prompt 
            
        prompt = template.format(
            # height=self.img_height,
            # width = self.img_width,
            prompt=self.prompt, characters=self.charcters)
        
        client = OpenAI(
            api_key = OPENAI_API_KEY, 
        )
        
        response = client.chat.completions.create(
                model = OPENAI_MODEL,
                messages =[
                    {"role": "user", "content" : f"""{prompt}"""}
                ],  
        )
        res =  response.choices[0].message.content
        
        if 'background' in res.lower():
            res = res.splitlines()[0]
        
        bboxes = eval(res)
        out_bboxs = dict()
        H, W = float(512), float(512)
        
        for (class_name, coord) in bboxes:
            x1, y1, box_width, box_height = coord
            
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            x1, x2 = np.divide((x1,x2), W)
            y1, y2 = np.divide((y1,y2), H)
            
            x1 *= self.img_width
            x2 *= self.img_width
            y1 *= self.img_height
            y2 *= self.img_height
            
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            
            out_bboxs[class_name]= [x1, x2, y1, y2]
    
        return out_bboxs

def get_prompt(txt):
    # str로부터 캐릭터 이름, scene_number, prompt 추출
    # character_name = txt[txt.find('-')+1:txt.find(':')].strip()
    scene_number = txt[txt.find(':')].strip()
    prompt = txt[txt.find(':')+1:].strip()
    
    return scene_number, prompt

def create_compositions(args, 
                        text2box,
                        # prompt, 
                        characters, 
                        # num, 
                        min_num_objects:int = 1):
    H, W = args.height, args.width
    
    
        
    # =================
    # Get bounding box
    # -----------------
    res = text2box.get_layout()
    
    classes = list(res.keys())
    bboxes = list(res.values())
    
    # ==============
    # Get obj. ratios
    # --------------
    
    if args.with_obj_ratio:
        object_ratios = text2box.obj_size_reactions()
        object_ratios = list(map(lambda string: string.lstrip(), object_ratios.split(',')))
        object_relations = {obj_name: float(object_ratios[u_idx])
            for u_idx, obj_name in enumerate(classes)}
        print('obj ratios')
        
        biggest_obj_name, _ = get_objects(object_relations)
    
    # out of range는 추가 못했삼 # 처리 필요
    
    # Paste largest object first, and then paste other smaller objects
    ious = box_iou(torch.Tensor(bboxes), torch.Tensor(bboxes))
    ious = ious * (1 - torch.eye(*ious.shape))
    ious = ious.sum(dim=-1)
    paste_order = torch.argsort(ious, descending=True).numpy()

    invalid = False
    
    # Preprocessing
    all_boxes_area: list = list()
    for obj_idx in paste_order:
        class_name = classes[obj_idx]
        
        llm_box_x1, llm_box_y1, llm_box_x2, llm_box_y2 = bboxes[obj_idx]
        
        llm_box_cent_x = llm_box_x1 + (llm_box_x2 - llm_box_x1) / 2.
        llm_box_cent_y = llm_box_y1 + (llm_box_y2 - llm_box_y1) / 2.
        
        if args.with_obj_ratio:
        # 추가를 못했어요
            biggest_obj_idx = classes.index(biggest_obj_name)
            biggest_obj_bbox = bboxes[biggest_obj_idx]
            llm_box_big_x1, llm_box_big_y1, llm_box_big_x2, llm_box_big_y2 = biggest_obj_bbox

            llm_box_big_width = (llm_box_big_x2 - llm_box_big_x1) * object_relations[class_name]
            llm_box_big_height = (llm_box_big_y2 - llm_box_big_y1) * object_relations[class_name]

            llm_box_x1 = int(llm_box_cent_x - llm_box_big_width / 2.)
            llm_box_x2 = int(llm_box_cent_x + llm_box_big_width / 2.)
            llm_box_y1 = int(llm_box_cent_y - llm_box_big_height / 2.)
            llm_box_y2 = int(llm_box_cent_y + llm_box_big_height / 2.)
            
        # Adjust x coordinates
        if llm_box_x1 < 0:
            llm_box_x2 += llm_box_x1  # Adjust llm_box_x2 left by the amount llm_box_x1 is out of bounds
            llm_box_x1 = 0
        if llm_box_x2 > W - 1:
            llm_box_x1 = max(llm_box_x1 - (llm_box_x2 - (W - 1)), 0)  # Adjust x1 without going out of bounds
            llm_box_x2 = W - 1

        # Adjust y coordinates
        if llm_box_y1 < 0:
            llm_box_y2 += llm_box_y1  # Adjust llm_box_y2 up by the amount llm_box_y1 is out of bounds
            llm_box_y1 = 0
        if llm_box_y2 > H - 1:
            llm_box_y1 = max(llm_box_y1 - (llm_box_y2 - (H - 1)), 0)  # Adjust y1 without going out of bounds
            llm_box_y2 = H - 1

        llm_box_width, llm_box_height = abs(llm_box_x2 - llm_box_x1), abs(llm_box_y2 - llm_box_y1)     

        all_boxes_area.append(llm_box_width * llm_box_height)

    min_box_area = np.min(all_boxes_area)
    min_box_area_in_canvas: float = (H / 5) * (W / 5) if len(characters) <= 3 else \
                                (H / 6) * (W / 6) if len(characters) == 4 else (H / 8) * (W / 8)
    if min_box_area < min_box_area_in_canvas:
        box_canvas_ratio: float = min_box_area_in_canvas / min_box_area
    else:
        box_canvas_ratio: float = 1.

    box_canvas_ratio: list = [box_canvas_ratio] * len(classes) # expand

    invalid = False
    final_bbox = dict()
    for obj_idx in paste_order:
        try:
            class_name, region = characters[obj_idx], bboxes[obj_idx]
        except:
            invalid = True
            print(f'[Invalid] Retrieved numbers of boxes are {len(bboxes)} while given number of objects are {len(classes)}')
            break
        
        # Get bounding boxes upper-left and lower-right coordinates from LLM model
        llm_box_x1, llm_box_y1, llm_box_x2, llm_box_y2 = region

        # Get bounding boxes centers, where boxes are generated by LLM model
        llm_box_cent_x = llm_box_x1 + (llm_box_x2 - llm_box_x1) / 2.
        llm_box_cent_y = llm_box_y1 + (llm_box_y2 - llm_box_y1) / 2.

        # Adjust x coordinates
        if llm_box_x1 < 0:
            llm_box_x2 += llm_box_x1  # Adjust llm_box_x2 right by the amount llm_box_x1 is out of bounds
            llm_box_x1 = 0
        if llm_box_x2 > W - 1:
            llm_box_x1 = max(llm_box_x1 - (llm_box_x2 - (W - 1)), 0)  # Adjust x1 without going out of bounds
            llm_box_x2 = W - 1

        # Adjust y coordinates
        if llm_box_y1 < 0:
            llm_box_y2 += llm_box_y1  # Adjust llm_box_y2 down by the amount llm_box_y1 is out of bounds
            llm_box_y1 = 0
        if llm_box_y2 > H - 1:
            llm_box_y1 = max(llm_box_y1 - (llm_box_y2 - (H - 1)), 0)  # Adjust y1 without going out of bounds
            llm_box_y2 = H - 1

        llm_box_width, llm_box_height = abs(llm_box_x2 - llm_box_x1), abs(llm_box_y2 - llm_box_y1)

        llm_box_width *= box_canvas_ratio[obj_idx]
        llm_box_height *= box_canvas_ratio[obj_idx]

        if llm_box_width==0.0:
            llm_box_width = 10
        if llm_box_height==0.0:
            llm_box_height = 10
            
        print(f'width : {llm_box_width}, height : {llm_box_height}')
        
        # llm_box_width, llm_box_height = int(llm_box_width), int(llm_box_height)

        
        # Maintain the original aspect ratio
        # src_obj_width, src_obj_height are the dimensions of the source object
        # llm_box_width, llm_box_height are the initial dimensions of the bounding box

        # Adjust bounding box dimensions based on the aspect ratio of the source object
        bbox= dict()
        
        bbox['x1'] = int(llm_box_x1)
        bbox['x2'] = int(llm_box_x2)
        bbox['y1'] = int(llm_box_y1)
        bbox['y2'] = int(llm_box_y2)
        
        
    
              
        final_bbox [f'{class_name}'] = bbox
  
    return final_bbox

if __name__ =='__main__':
    args = parse_args()
    Path(args.save_path).mkdir(exist_ok=True, parents=True)
    
    with open(args.input_characters, 'r') as characters_json:
        characters = json.load(characters_json)

    f = open(args.scene_path, 'r')
    full_txt = []
    
    for line in f.readlines():
        if line =='\n':
            pass
        else:
            full_txt.append(line)

    layout = dict()
    for num, txt in enumerate(full_txt):
        scene_number, prompt = get_prompt(txt)
        
        input_characters = ', '.join(list(characters.values()))
        input_characters_name = list(characters.keys())
        
        # text2box=None
        text2box = Text2Box(prompt, input_characters, img_height=args.height, img_width=args.width)
        # _layout = text2box.get_layout()
        _layout = create_compositions(args=args, text2box=text2box, characters=input_characters_name)
        
        layout[f'img_{num+1}.png'] = _layout
    
    print(layout)
    
    with open(args.save_path+'/layout.json', 'w') as file:
        json.dump(layout, file, indent=4)
        # create_compositions(args=args, text2box=text2box, prompt=prompt, characters=characters, num=num)


    
