from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, AutoPipelineForInpainting
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, DDIMScheduler, AutoencoderKL, EulerDiscreteScheduler, DPMSolverMultistepScheduler

from PIL import Image, ImageFilter
from skimage import exposure
from pathlib import Path

from rembg import remove
import torch
import numpy as np
import cv2
import argparse
import json
import os

from template import styles
# python generate_scene.py --layout_path output/full_story2/Lemi+Cindy.json --prompt_path output/full_story2/background.txt --input_characters Lemi Cindy --scene_style Disney --characters_path output/mask/

def parse_args():
    parser = argparse.ArgumentParser(description='Generating scene')
    parser.add_argument('--layout_path', default='./output/story/layout.json', help='Enter your character layout path .json format')
    parser.add_argument('--characters_path', default='./output/mask', help='character images with mask')
    # parser.add_argument('--input_characters', nargs="+",
    #                     default=[], 
    #                     help='Enter your character name')
    parser.add_argument('--bkg_prompt_path', default="./output/story/background.txt", type=str, help="txt file path")
    parser.add_argument('--scene_prompt_path', default="./output/story/scene.txt", type=str, help="txt file path")
    parser.add_argument('--save_path', default='./output/image', help='saving path generated scene images')
    parser.add_argument('--refiner', default=True, help="applying refiner")
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
    parser.add_argument('--negative_prompt', default="character, ugly, person, man, woman", type=str, help="SDXL background negative prompt")
    parser.add_argument('--num_inference_steps', default=20, type=int, help="SDXL inference steps")
    parser.add_argument('--guidance_scale', default=10.0, type=float, help="SDXL guidance scale")
    parser.add_argument('--strength', default=0.99, type=float, help="SDXL strength")
    parser.add_argument('--seed', default=2572265, type=int, help="SDXL seed")
    parser.add_argument("--device", default="cuda:0", type=str, help="GPU device")
    parser.add_argument('--blur_size', default=5, type=int, help="blur filter size between characters and scenes. if you don't want to use blur option, put the size 0.")
    
    args = parser.parse_args()
    return args


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def apply_style(style_name, prompt, negative=""):
    p, n = styles.get(style_name, styles[style_name])
    p = p.replace("{prompt}", prompt)
    n = negative + n
    return p, n


def crop_image(image):
    gray_image = image.convert("L")
    
    canny = gray_image.filter(ImageFilter.FIND_EDGES)
    
    canny = np.array(canny, dtype=np.uint8)
    pts = np.argwhere(canny > 0)
    
    # pts = np.array(image, dtype=np.uint8)
    # pts = np.argwhere(pts>0)
    
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)

    return x1, y1, x2, y2

# def delete_bkg(mask, image):
#     mask = mask.convert("RGBA")
#     alpha_data = []
#     for pixel in mask.getdata():
#         if pixel[0] == 255 \
#             and pixel[1] == 255 \
#             and pixel[2] == 255:
#             alpha_data.append(0)
#         else:  
#             alpha_data.append(255)
#     alpha_channel = Image.new("L", mask.size)
#     alpha_channel.putdata(alpha_data)

#     new_image = image.convert("RGBA")
#     new_image.putalpha(alpha_channel)
#     return new_image
    
def make_masked_image(background, layout, image, mask=None):
    if mask is None :
        background.paste(image, layout)
    else:
        background.paste(image, layout, mask)
        
    return background

def make_bkg(args, prompt, device="cuda:0"):
    negative_prompt = args.negative_prompt 
    
    # style apply option
    if args.scene_style is not None:
        prompt, negative_prompt = apply_style(args.scene_style, prompt=prompt,negative=negative_prompt)
    
    back_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to(device)
    
    print(f"🚗🚗🚗🚗🚗 prompt : {prompt}, \n negative prompt : {negative_prompt}")
    
    output = back_pipe(prompt=prompt)
    # output.images[0].save("./test_bk3.jpg") # check
    path = args.save_path +'/test_bkg.png'
    output.images[0].save(path)
    return output.images[0]

def make_scene(args, prompt, image, mask_image, save_path, device="cuda:0"):
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
        torch_dtype=torch.float16, 
        use_safetensors=True,
        variant="fp16"
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    refiner = AutoPipelineForInpainting.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16, 
        use_safetensors=True,
        variant="fp16"
    ).to(device)
    
    if args.scene_style is not None:
        prompt, negative_prompt = apply_style(args.scene_style, prompt=prompt)
        
    output = pipe(
        prompt = prompt,
        negative_prompt = negative_prompt,
        image = image,
        mask_image = mask_image,
        guidance_scale = args.guidance_scale,
        num_inference_steps = args.num_inference_steps,
        strength = args.strength,
        seed=args.seed,
        # denoising_start=0.8
    )
    
    if args.refiner == True :
        output = refiner(
            prompt=prompt,
            negative_prompt = negative_prompt,
            image=output.images,
            mask_image=mask_image,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            denoising_start=0.85
        )    
    output.images[0].save(save_path)

if __name__ == '__main__':
    args=parse_args()
    Path(args.save_path).mkdir(exist_ok=True, parents=True)
    
    # text file
    f = open(args.bkg_prompt_path, "r")
    bkg_prompts = []
    
    for line in f.read().split("\n\n"):
    # for line in f.read().split("\n"):
        bkg_prompts.append(line)
        
    if Path(args.scene_prompt_path) is True:
        f = open(args.scene_prompt_path, "r")
        scene_prompts = []
        for line in f.read().split("\n\n"):
            scene_prompts.append(line)
    else:
        scene_prompts = bkg_prompts
    
    with open(args.layout_path) as f:
        layout = json.load(f)

    
    # Generated scenes 
    for lay, prompt in enumerate(bkg_prompts): # 원래코드
    # for lay in range(6,8):
        layout_key = 'img_'+str(lay+1)+'.png'
        
        background = make_bkg(args=args, prompt=bkg_prompts[lay])
        width, height = background.size
        mask_background = Image.new('RGB', (width, height), (0,0,0))
        
        # print(args.input_characters, layout[layout_key].keys())
        
        for character in layout[layout_key].keys(): #args.input_characters:
            image_path = args.characters_path + '/' + str(lay+1) + '_' + character +'_foreground.png'
            mask_image_path = args.characters_path + '/' + str(lay+1) + '_' + character +'_mask.png'
            
            if os.path.isfile(image_path) is True:  
                image = Image.open(image_path).resize((1024,1024))
                mask_image = Image.open(mask_image_path).resize((1024,1024)) 
                
                size_x = abs(layout[layout_key][character]['x1'] - layout[layout_key][character]['x2'])
                size_y = abs(layout[layout_key][character]['y1'] - layout[layout_key][character]['y2'])
                
                image = image.resize((size_x,size_y)) 
                mask_image = mask_image.resize((size_x,size_y))
                print(f"✔✔✨{character} : {size_x}, {size_y}")
                
                x = min(layout[layout_key][character]['x1'], layout[layout_key][character]['x2'])
                y = min(layout[layout_key][character]['y1'], layout[layout_key][character]['y2'])
                    
                # # crop foreground mask
                # x1, y1, x2, y2 = crop_image(mask_image)
                
                # img = np.array(mask_image, dtype=np.uint8)
                # img = np.argwhere(img > 0)
                # y1, x1 = img.min(axis=0)
                # y2, x2 = img.max(axis=0)
                
                # print(x1, x2, y1, y2)
                
                # image = image.crop((min(x1, x2), min(y1, y2), abs(x1-x2), abs(y1-y2)))
                # mask_image = mask_image.crop((min(x1, x2), min(y1, y2), abs(x1-x2), abs(y1-y2)))
                
                alpha_mask = mask_image.convert("L")
                
                make_masked_image(background=background, layout=(x,y), image=image, mask=alpha_mask)
                
                make_masked_image(background=mask_background, layout=(x,y), image=mask_image, mask=alpha_mask)
                
        background.save(args.save_path+f'/bkg_{lay+1}.png')
        mask_background.save(args.save_path+f'/mask_bkg_{lay+1}.png')
        # mask_background.save(args.save_path+f'/masked_bkg_{lay+1}.png')
        
        if args.blur_size > 0 :    
            blurred_img = mask_background.filter(ImageFilter.BoxBlur(args.blur_size)) # : 5

            blurred_img_array = np.array(blurred_img)
            stretched_img = exposure.rescale_intensity(
                                blurred_img_array,
                                in_range=(127.5, 255),
                                out_range=(0, 255))
            mask = Image.fromarray(np.uint8(stretched_img))
        
        mask = Image.fromarray(255 - np.array(mask_background))
        save_path = args.save_path+'/'+layout_key
        
        # print(background)
        print(f"make scene {lay+1} generating... ")
        # print(mask)
        make_scene(args=args, prompt=scene_prompts[lay], image=background, mask_image=mask, save_path=save_path)
    
    
    