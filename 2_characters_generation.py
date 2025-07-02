'''
*** 각 문단별로 진행해야함
캐릭터 생성하기 : txt파일로부터 입력 -> 캐릭터 
캐릭터가 이미지에 가득 차도록 crop하기
segmentation map foreground, mask 만들기


예시 : Scene 1-John: John is a small mouse who lives behind a pantry. He is skittish and cautious, always on the lookout for predators like Cindy, the sleek cat known for hunting.
'''

import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, ControlNetModel, StableDiffusionXLPipeline
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from PIL import Image
from pathlib import Path

from ip_adapter import IPAdapter, IPAdapterPlusXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
from diffusers.utils import load_image
from transformers import pipeline
from controlnet_aux import OpenposeDetector

import cv2
import argparse
# from segment_character import run_image
import json
import re
import random

from template import styles

def parse_args():
    parser = argparse.ArgumentParser(description="Custom character editing")
    parser.add_argument("--img_path", type=str, default="input/face", help="source characters images folder path")
    parser.add_argument("--save_path", type=str, default="output/characters", help="save output character images folder path")
    parser.add_argument("--prompt_path", type=str, default="output/story/characters.txt", help="prompt txt file path")
    parser.add_argument("--bkg_path", type=str, default="output/story/background.txt", help="background txt file path")
    parser.add_argument('--negative_prompt', default="ugly, distortion, multiple people, many, ((multiple characters)), (((overshadowed))), background, pattern, distorted face and body", type=str, help="SDXL background negative `prompt")
    
    parser.add_argument('--input_characters', type=str, default="output/story/characters.json", help="character names and features")
    
    parser.add_argument('--scene_style',default='',
                        choices=['',
                            'Japanese Anime', 
                            'Cinematic', 
                            'Disney', 
                            'Photographic',
                            'Comics', 
                            'Line'
                            ],
                        help="generation image style"
                        )
    
    parser.add_argument("--base_model_path", default="stabilityai/stable-diffusion-xl-base-1.0", type=str)
    parser.add_argument("--image_encoder_path", default="IP-Adapter/models/image_encoder", type=str, help="IPAdapter encoder ckpt path")
    parser.add_argument("--ip_ckpt", default="IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin", type=str, help="IPAdapter ckpt path")
    parser.add_argument('--seed', default=42, type=int, help="img generate seed")
    
    
    args = parser.parse_args()
    return args

def get_prompt(txt):
    # str로부터 캐릭터 이름, scene_number, prompt 추출
    character_name = txt[txt.find('-')+1:txt.find(':')].strip()
    scene_number = txt[:txt.find('-')].strip()
    prompt = txt[txt.find(':')+1:].strip()
    
    return character_name, scene_number, prompt

def generate_image(prompt):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
    ).to("cuda")
    
    image = pipeline(prompt)
    
    return image.images[0]

def get_kepoints_image(image):
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    return openpose(image)

# editing 안쓴다.
def editing(character, keypoints):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16 )
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, vae=vae, feature_extractor = None, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    
    ip_model = IPAdapter(pipe, "IP-Adapter/models/image_encoder", "IP-Adapter/models/ip-adapter_sd15.bin", "cuda")
    generate_image = ip_model.generate(pil_image=character, image=keypoints, num_samples=1, num_inference_steps=30)
    
    print(generate_image)
    return generate_image[0]

def apply_style(style_name, prompt, negative=""):
    p, n = styles.get(style_name, styles[style_name])
    p = p.replace("{prompt}", prompt)
    n = negative + n
    return p, n

class StoryHM():
    def __init__(
        self, 
        img_path, 
        base_model_path, 
        image_encoder_path, 
        ip_ckpt, 
        prompt, 
        negative_prompt,
        style,
        seed,
        device = "cuda",
        num_inference_steps=30,
        guidance_scale = 7.5 
        ):
        
        self.img_path = img_path
        self.base_model_path = base_model_path
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.style = style
        
        self.device = device
        self.num_inference_steps=num_inference_steps
        self.seed = seed
        self.guidance_scale = guidance_scale
        
        # self.pipe
        
    
    def set_base_model(self):
        pipe = StableDiffusionXLCustomPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype = torch.float16,
            add_watermarker=False
        )  
        return pipe
    def generate(self, pipe):
        ip_model = IPAdapterPlusXL(pipe, self.image_encoder_path, self.ip_ckpt, self.device, num_tokens=16)
        
        image = Image.open(self.img_path)
        images = ip_model.generate(
            pil_image = image,
            num_samples=1,
            num_inference_steps=self.num_inference_steps,
            # seed=self.seed,
            prompt = self.prompt,
            negative_prompt = self.negative_prompt
            # guidance_scale=self.guidance_scale
        )
        return images[0]
    
def run_image(args):
    f = open(args.prompt_path, 'r')
    full_txt = []
    
    for line in f.readlines():
        if line =='\n':
            pass
        else:
            full_txt.append(line)
    
    f = open(args.bkg_path, 'r')
    bkg_txt = []
    for line in f.readlines():
        if line =='\n' :
            pass
        else:
            bkg_txt.append(line)
    
    for txt in full_txt:
        # print(txt)
        with open(args.input_characters, 'r') as characters_json:
            features = json.load(characters_json)
        character_name, scene_number, prompt = get_prompt(txt)
        
        prompt, negative_prompt = apply_style(args.scene_style, prompt=prompt, negative=args.negative_prompt)
        
        
        print(scene_number+'------------------------'+character_name)
        prompt = f"White background, portrait, (((((single character))))), {character_name} {features[character_name]} is {prompt}"
        prompt = prompt.replace(';', ', ')
        print(prompt)
        
        custom_character_path = args.img_path + '/' + character_name +'_face.png'
        
        crt = StoryHM(img_path=custom_character_path, 
                  base_model_path=args.base_model_path,
                  image_encoder_path=args.image_encoder_path,
                  ip_ckpt=args.ip_ckpt,
                  prompt=prompt,
                  negative_prompt=negative_prompt,
                  style=args.scene_style,
                  seed=args.seed
                  )
        base = crt.set_base_model()
        character = crt.generate(pipe=base)
        
        save_path = args.save_path + '/' + scene_number +'_'+character_name+'.png'
        character.save(save_path)
        
        


if __name__ == '__main__':
    args = parse_args()
    
    Path(args.save_path).mkdir(exist_ok=True, parents=True)
    
    run_image(args=args)
    
    
    # 기존 방식
    # character = generate_image(prompt=prompt)
    # keypoints = get_kepoints_image(character)
    
    # custom_character_path = args.img_path + '/' + character_name +'.jpg'
    # custom_character = Image.open(custom_character_path)
    # custom_character = editing(custom_character, keypoints)
    
    # save_path = args.save_path + '/' + scene_number +'_'+character_name+'.jpg'
    # custom_character.save(save_path)
    
    
        # change_list = ['he ', 'He ', 'his ', 'His ', 'she ', 'She ', 'her ', 'Her ']
        
        # for value in change_list:
        #     prompt = prompt.replace(value, character_name+' ')
        # prompt = re.sub(character_name, features[character_name], prompt)
        
        # for crt in features.keys():
        #     prompt = prompt.replace(crt, "")