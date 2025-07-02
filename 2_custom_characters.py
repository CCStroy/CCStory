import torch
from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline, StableDiffusionControlNetPipeline, StableDiffusionPipeline, AutoPipelineForImage2Image, StableDiffusionXLImg2ImgPipeline, DiffusionPipeline
from diffusers import DDIMScheduler, UniPCMultistepScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers import ControlNetModel, AutoencoderKL

from ip_adapter import IPAdapter, IPAdapterPlusXL, IPAdapterPlus
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceIDXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline

from diffusers.utils import load_image
from transformers import pipeline
from controlnet_aux import OpenposeDetector
from PIL import Image
from pathlib import Path

from template import styles, style_model, negative_prompt
from insightface.app import FaceAnalysis
import cv2
import argparse
# from segment_character import run_image
import json
import re
import random
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def parse_args():
    parser = argparse.ArgumentParser(description="Custom character editing")
    parser.add_argument("--img_path", type=str, default="input/face", help="source characters images folder path")
    parser.add_argument("--save_path", type=str, default="output/characters", help="save output character images folder path")
    parser.add_argument("--prompt_path", type=str, default="output/story/characters.txt", help="prompt txt file path")
    # parser.add_argument("--bkg_path", type=str, default="output/story/background.txt", help="background txt file path")
    # parser.add_argument("--keywords")
    parser.add_argument('--input_characters', type=str, default="output/story/characters.json", help="character names and features")
    
    
    parser.add_argument('--model', default='sdxl-base',
                        choices=[
                            'sdxl-base',
                            'sd1.5-base',
                            'sd1.5-anime',
                            'sdxl-img2img', 
                            'test',
                            'flintstone'
                        ], help="character generation diffusion models")
    parser.add_argument("--seed", default=1234, type=int, help="diffusion model seed")
    
        
    args = parser.parse_args()
    return args
    
sd_character_dict ={
    "sdxl-base":dict(
        sd_pipe = StableDiffusionXLPipeline,
        sd_pipe_cnet = StableDiffusionXLControlNetPipeline,
        ip_pipe = IPAdapterPlusXL
    ),
    "sdxl-img2img":dict(
        sd_pipe = StableDiffusionXLImg2ImgPipeline,
        sd_pipe_cnet = StableDiffusionXLControlNetPipeline,
        ip_pipe = IPAdapterPlusXL
    ), # 안됨
    "sd1.5-base":dict(
        sd_pipe = StableDiffusionPipeline,
        sd_pipe_cnet = StableDiffusionControlNetPipeline,
        ip_pipe = IPAdapterPlus
    ),
    "sd1.5-anime":dict(
        sd_pipe = StableDiffusionPipeline,
        sd_pipe_cnet = StableDiffusionControlNetPipeline,
        ip_pipe = IPAdapterPlus
    ),
    "test":dict(
        sd_pipe = AutoPipelineForImage2Image,
        sd_pipe_cnet = StableDiffusionXLControlNetPipeline,
        ip_pipe = IPAdapterPlusXL
    ),
    "flintstone":dict(
        sd_pipe = StableDiffusionXLPipeline,
        sd_pipe_cnet = StableDiffusionXLControlNetPipeline,
        ip_pipe = IPAdapterPlusXL
    )
}

class StoryHM():
    def __init__(
        self,
        face_img_path, 
        prompt, 
        negative_prompt,
        seed,
        model,
        body_img_path = None,
        device = "cuda:0",
        num_inference_steps=30,
        guidance_scale = 7.5, 
        num_tokens = 16,
        cnet_ratio = 0.25
        ):
        
        self.model = model
        self.face_img_path = face_img_path
        self.body_img_path = body_img_path
        
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        
        self.device = device
        self.num_inference_steps=num_inference_steps
        self.seed = seed
        self.guidance_scale = guidance_scale
        self.num_tokens= num_tokens
        self.cnet_ratio = cnet_ratio
        
    def base_image(self):
        base_model :str = style_model[self.model]['base_model']
        vae_model_path :str = style_model[self.model]['vae_model_path']
        
        vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=torch.float16)      
        
        pipe = sd_character_dict[self.model]["sd_pipe"].from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        ).to(self.device)        
        
        if style_model[self.model]["lora_weights"] is not None:
            pipe.load_lora_weights(style_model[self.model]["lora_weights"])
        
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        
        # if sd_character_dict[self.model]["sd_pipe"] == "sdxl-img2img":
        # if self.body_img_path is not None:
        #     body = load_image(self.body_img_path).convert("RGB")
        #     image = pipe(prompt= self.prompt, image=body, negative_prompt=self.negative_prompt)
        # else :
        image = pipe(prompt= self.prompt, negative_prompt=self.negative_prompt)
        
        return image.images[0]
    
    def generate(self, depth_img):
        base_model = style_model[self.model]["model"]
        vae_model_path = style_model[self.model]["vae_model_path"]
        image_encoder_path = style_model[self.model]["image_encoder_path"]
        ip_ckpt = style_model[self.model]["ip_ckpt"]
        image = Image.open(self.face_img_path)
        
        vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=torch.float16)
        
        controlnet = ControlNetModel.from_pretrained(style_model[self.model]["controlnet"], torch_dtype=torch.float16)
        
        pipe_cnet = sd_character_dict[self.model]["sd_pipe_cnet"].from_pretrained(
            base_model,
            controlnet = controlnet,
            torch_dtype = torch.float16,
            vae=vae,
            feature_extractor = None,
            safety_checker = None
        ).to(self.device)
        
        pipe_cnet.scheduler = DPMSolverMultistepScheduler.from_config(pipe_cnet.scheduler.config)
        
        # if style_model[self.model]["lora_weights"] is not None:
        #     # pipe_cnet.load_lora_weights("ip-adapter-faceid_sd15_lora.safetensors")
            
        #     ip_model = sd_character_dict[self.model]["ip_pipe"](
        #         pipe_cnet,
        #         # image_encoder_path,
        #         ip_ckpt,
        #         self.device,
        #         # num_tokens=self.num_tokens
        #     )
            
        #     app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        #     app.prepare(ctx_id=0, det_size=(640, 640))

        #     image = cv2.imread(self.face_img_path)
        #     faces = app.get(image)
            
        #     faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        #     images = ip_model.generate(
        #         faceid_embeds = faceid_embeds,
        #         # image=depth_img,
        #         num_samples = 1,
        #         width=1024,
        #         height=1024,
        #         prompt = self.prompt,
        #         negative_prompt = self.negative_prompt,
        #         seed=self.seed,
        #         num_inference_steps = self.num_inference_steps,
        #         controlnet_conditioning_scale = self.cnet_ratio,
        #     )
            
        # else :       
        ip_model = sd_character_dict[self.model]["ip_pipe"](
            pipe_cnet,
            image_encoder_path,
            ip_ckpt,
            self.device,
            num_tokens=self.num_tokens
        )
    
        images = ip_model.generate(
            pil_image = image,
            image=depth_img,
            num_samples =1,
            prompt = self.prompt,
            negative_prompt = self.negative_prompt,
            seed=self.seed,
            num_inference_steps = self.num_inference_steps,
            controlnet_conditioning_scale = self.cnet_ratio
        )
        
        return images[0]
    
def get_prompt(txt):
    # str로부터 캐릭터 이름, scene_number, prompt 추출
    character_name = txt[txt.find('-')+1:txt.find(':')].strip()
    scene_number = txt[:txt.find('-')].strip()
    prompt = txt[txt.find(':')+1:].strip()
    
    return character_name, scene_number, prompt

def apply_style(style_name, prompt, negative=""):
    p, n = styles.get(style_name, styles[style_name])
    p = p.replace("{prompt}", prompt)
    n = negative + n
    return p, n

def depth_map_extracter(image):
    # image = Image.frombytes("RGBA", ((512,512)),image)
    depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
    depth_image = depth_estimator(image)['depth']
    
    return depth_image


def run_image(args,
                ext: str = 'png'):
    f = open(args.prompt_path, 'r')
    full_txt = []
    
    for line in f.readlines():
        if line =='\n':
            pass
        else:
            full_txt.append(line)
    with open(args.input_characters, 'r') as characters_json:
        features = json.load(characters_json)
    
    # def describe_character(path):
    # captioner = pipeline("image-to-text",
    #                      model="Salesforce/blip-image-captioning-large")
    # return captioner(path)[0]['generated_text']
    
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
            
    for txt in full_txt:
        character_name, scene_number, prompt = get_prompt(txt)
        custom_character_face = f"{args.img_path}/{character_name}_face.{ext}"
        custom_character_body = f"{args.img_path}/{character_name}_body.{ext}"
        
        print(scene_number+'------------------------'+character_name)
        # caption = captioner(custom_character_body)[0]['generated_text']
        
        if character_name == "Fred":
            caption = "black hair, blue scarf, orange onepiece"
        elif character_name == "Wilma":
            caption = "red hair, single hair bun, white dress, white perl neckless,"
        elif character_name == "Jobs":
            caption = "black turtleneck"
        # elif character_name == "Musk":
        #     caption = "full body"
        else :
            caption = ""
        
        prompt = f"{character_name} is {prompt}, high quality, no background, full body, {caption}" 
        # prompt = f"{caption} {features[character_name]} is {prompt}, high quality, no background" 
        
        # prompt = f"high quality, highly detailed, full body, a girl in a blue dress with a red bow, {prompt}" 
        # White background, {character_name} {features[character_name]} is 
        prompt = prompt.replace(';', ',')
        print(prompt)
        
        
        crt = StoryHM(
            face_img_path=custom_character_face, 
            body_img_path=custom_character_body,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=args.seed,
            model=args.model,
            cnet_ratio = 0.45)
            
        
        base_image = crt.base_image()
        depth_image = depth_map_extracter(base_image)
        base_image.save(f"{args.save_path}/{character_name}_base_{scene_number}.{ext}")
        character = crt.generate(depth_img=depth_image)
        
        save_path = f"{args.save_path}/{scene_number}_{character_name}.{ext}"
        character.save(save_path)
        

if __name__ == '__main__':
    args = parse_args()
    Path(args.save_path).mkdir(exist_ok=True, parents=True)
    run_image(args=args)