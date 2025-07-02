# story generation
# ==========
# input prompt template
# ==========
story_template = '''Never use 'he', 'she', 'it', or 'they' in the story.  Do not call subjects in general like using 'a thing', 'a person', 'they', 'a girl', 'the trio'. Make sure when you describe the subjects, you must use their names. Make sure to create a story with only the given characters. YOU MUST GENERATE ABOUT ONLY {characters}'''

scene_template = '''Split the above story to sentences, each paragraph corresponds to a sentence. 
Each summary should start with a 'Scene [scene number]:'. And you must clarify the name of characters clearly on each scene. Do not call subjects in general like using 'a person', 'they', 'a girl', 'the trio'. Make sure in each scene when you describe the subjects, you must use their names.'''

# character_template = '''Generate the characteristics of each character from each scene. 
# In character descriptions, describe in as much detail as possible, such as facial expressions, movements, etc. especially, write about facial expressions and emotions. The character description within begins with '[scene number]-[character name]:'. 

# Never use 'he', 'she', 'it', or 'they' in the scene descriptions. Do not call subjects in general like using 'a thing', 'a person', 'they', 'a girl', 'the trio'. Never have more than one character in the scene description.
# Make sure when you describe the subjects, you must use their names. 

# If there are more than one character, you must divide character descriptions from same scene. 
# For example, if Sally and John both appear in 3, write 3-Sally :~, 3-John :~ separately. The scenes are as follow : '''
# Never use 'he', 'she', 'it', or 'they' in the words. Do not call subjects in general like using 'a thing', 'a person', 'they', 'a girl', 'the trio'. Do not mention other characters each description.

character_template = '''
Describe the {characters} facial expressions and body actions in words in detail from each scene. within begins with '[scene number]-[character name]:'. You only use WORDS. DO NOT MAKE A SENTENCE.

If there are more than one character in a scene, MUST divide character descriptions from same scene. You must create each character detail in one scene. However, if the scene does not contain all characters, you must generate only the characters that are included.

YOU MUST WRITE ONLY ONE CHATACTER IN ONE SCENE.
YOU MUST WRITE ONLY ONE CHATACTER IN ONE SCENE.
YOU MUST WRITE ONLY ONE CHATACTER IN ONE SCENE.
YOU MUST WRITE ONLY ONE CHATACTER IN ONE SCENE.

YOU MUST GENERATE ABOUT ONLY {characters}
YOU MUST GENERATE ABOUT ONLY {characters}
YOU MUST GENERATE ABOUT ONLY {characters}

For example,
    1-Kiki : calm, soft smile, steady gaze
'''

# For example,
#     1-Kiki: calm contentment, tranquility, relaxed eyelids, soft smile, steady gaze
#     1-Zizi : triumphant cheer, radiant eyes, overjoyed grunts, unabashed excitement, exchanging glances 
    
#     2-Zizi: alert stance, bristling fur, low growls, eying surrounding cautiously, fearful of unknown sounds 
    
#     3-Kiki: keen observation, wide-eyed curiosity, calculating each leap, learning from Zizi, understanding dawns slowly
#     3-Zizi: flickering green eyes, lively movements, rubbing against legs, sleepiness seeping in, comfortable on lap

background_template = '''You are a prompter for background place image generation using Stable Diffusion. Write a location description very in detail. I give you each scene scripts start with "[scene number]". you make a prompt for background image generation. Never use 'he', 'she', 'it', or 'they', 'a girl', 'a cat', 'a mouse', character names, in the scene descriptions. 
The generated background prompts have a format,
    "[scene number] : A [adjective] [location] with [key elements]. [Additional details or specific items],."
    
The locations examples are 'garden', 'room', 'office', 'town', 'park', 'living room', 'bed room', 'farm', 'playroom' etc..

The scene scripts are as follow : '''

box_template = '''You are an intelligent bounding box generator.
    I will provide you with a caption for a photo, image, or painting.
    Your task is to generate the bounding boxes for the objects mentioned in the caption, along with a background prompt describing the scene.
    The images are of size 512x512, and the bounding boxes should not overlap or go beyond the image boundaries.
    Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]) and include exactly one object.
    Make the boxes larger if possible.
    Do not put objects that are already provided in the bounding boxes into the background prompt.
    If needed, you can make reasonable guesses.
    Generate the object descriptions and background prompts in English even if the caption might not be in English.
    Do not include non-existing or excluded objects in the background prompt. Please refer to the example below for the desired format.
    Please note that a dialogue box is also an object.
    MAKE A REASONABLE GUESS OBJECTS MAY BE IN WHAT PLACE.
    The top-left x coordinate + box width MUST NOT BE HIGHER THAN 512.
    The top-left y coordinate + box height MUST NOT BE HIGHER THAN 512.
    
    Caption: A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky
    Characters : car, truck, ballon, bird
    Objects: [('car', [21, 181, 211, 159]), ('truck', [269, 181, 209, 160]), ('balloon', [66, 8, 145, 135]), ('bird', [296, 42, 143, 100])]

    Caption: A watercolor painting of a wooden table in the living room with an apple on it
    Characters : table, apple
    Objects: [('table', [65, 243, 344, 206]), ('apple', [206, 306, 81, 69])]

    Caption: A watercolor painting of two pandas eating bamboo in a forest
    Characters : panda, bamboo
    Objects: [('panda', [30, 171, 212, 226]), ('bambooo', [264, 173, 222, 221])]

    Caption: A realistic image of four skiers standing in a line on the snow near a palm tree
    Characters : skier, skier, skier, skier, palm tree
    Objects: [('skier', [5, 152, 139, 168]), ('skier', [278, 192, 121, 158]), ('skier', [148, 173, 124, 155]), ('palm tree', [404, 180, 103, 180])]

    Caption: An oil painting of a pink dolphin jumping on the left of a steam boat on the sea
    Characters : steam boat, dolphin
    Objects: [('steam boat', [232, 225, 257, 149]), ('dolphin', [21, 249, 189, 123])]

    Caption: A realistic image of a cat playing with a dog in a park with flowers
    Characters : cat, dog
    Objects: [('cat', [51, 67, 271, 324]), ('dog', [302, 119, 211, 228])]
    
    ENSURE that generated bounding boxes NOT overlapped with each other.
    ENSURE that generated bounding boxes NOT overlapped with each other.
    ENSURE that generated bounding boxes NOT overlapped with each other.
    '''
box_prompt = '''
    Caption: {prompt}.
    Characters : {characters}
    Objects: '''

ratio_template = '''
    You are an intelligent object ratio generator.
    I will provide you with several object names.
    Your task is to generate the reasonable ratio for objects in the real world.
    The ratio of the biggest object equals to 1.
    The ratio of the smallest object equals to 0.4.

    Objects: house, person, pig, cow, car
    Ratio: 1.0, 0.5, 0.40, 0.55, 0.85

    Caption: bus, car, car, phone box
    Ratio: 1.0, 0.7, 0.7, 0.6

    Objects: cat, dog
    Ratio: 0.7, 1.0

    Objects: car, restroom
    Ratio: 0.8, 1.0

    Objects: cat, dog, house plant
    Ratio: 0.8, 1.0, 0.75

    Objects: house plant, fridge
    Ratio: 0.6, 1.0
    
    '''    
    
ratio_prompt = '''
    Objects: {prompt}.
    Ratio:'''

style_model ={
    "sdxl-base":dict(
        base_model = "stabilityai/stable-diffusion-xl-base-1.0",       
        model = "stabilityai/stable-diffusion-xl-base-1.0",       
        vae_model_path="madebyollin/sdxl-vae-fp16-fix",        
        image_encoder_path = "IP-Adapter/models/image_encoder",
        ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
        controlnet = "diffusers/controlnet-depth-sdxl-1.0",
        lora_weights = None
    ),
    "sdxl-img2img":dict(
        base_model = "stabilityai/stable-diffusion-xl-refiner-1.0",
        model = "stabilityai/stable-diffusion-xl-base-1.0",           
        vae_model_path="madebyollin/sdxl-vae-fp16-fix",        
        image_encoder_path = "IP-Adapter/models/image_encoder",
        ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
        controlnet = "diffusers/controlnet-depth-sdxl-1.0",
        lora_weights = None
    ),
    "sd1.5-base":dict(
        base_model="runwayml/stable-diffusion-v1-5",
        model="runwayml/stable-diffusion-v1-5",
        vae_model_path="stabilityai/sd-vae-ft-mse",
        image_encoder_path="IP-Adapter/models/image_encoder",
        ip_ckpt="IP-Adapter/models/ip-adapter-plus-face_sd15.bin",
        controlnet="lllyasviel/sd-controlnet-depth",
        lora_weights = None
    ),
    "sd1.5-anime":dict(
        base_model="dreamlike-art/dreamlike-anime-1.0",
        model="dreamlike-art/dreamlike-anime-1.0",
        vae_model_path="stabilityai/sd-vae-ft-mse",
        image_encoder_path="IP-Adapter/models/image_encoder",
        ip_ckpt="IP-Adapter/models/ip-adapter-plus-face_sd15.bin",
        controlnet="lllyasviel/sd-controlnet-depth",
        lora_weights = None
    ),
    "test":dict(
        base_model = "runwayml/stable-diffusion-v1-5",
        model = "stabilityai/stable-diffusion-xl-base-1.0",              
        vae_model_path="madebyollin/sdxl-vae-fp16-fix",        
        image_encoder_path = "IP-Adapter/models/image_encoder",
        ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
        controlnet = "diffusers/controlnet-depth-sdxl-1.0",
        lora_weights = None
    ),
    "flintstone":dict(
        base_model = "stabilityai/stable-diffusion-xl-base-1.0",
        model = "stabilityai/stable-diffusion-xl-base-1.0",
        vae_model_path="madebyollin/sdxl-vae-fp16-fix",        
        image_encoder_path = "IP-Adapter/models/image_encoder",
        ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
        controlnet = "diffusers/controlnet-depth-sdxl-1.0",
        lora_weights = "juliajoanna/sd-flintstones-model-lora-sdxl"
    )
}

negative_prompt =  '''
 ugly, Super Deformation character, distortion, bad anatomy, bad quality, blurry, cropped, error, extra hands, fused fingers, text, signature, multiple people, many, ((multiple characters)), (((overshadowed))), ((((((background)))))), pattern, distorted face, watermark, worst quality, out of frame, parts, objects, deformed, person, man, woman, NSFW
'''
# "ugly, distortion, bad anatomy, bad quality, blurry, cropped, error, extra hands, fused fingers, text, signature, multiple people, many, ((multiple characters)), (((overshadowed))), background, pattern, distorted face, watermark, worst quality, out of frame, parts, objects, deformed, hair accessory, ((ear rings)), ((hat)), ((NSFW)), "

# scene generation - prompting
style_list = [
    {
        "name": "",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Japanese Anime",
        "prompt": "anime artwork illustrating {prompt}. created by japanese anime studio, highly emotional, best quality, high resolution, background",
        "negative_prompt": "low quality, low resolution"
    },
    {
        "name": "Cinematic",
        "prompt": "{prompt}. ((cinematic)), emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy, background",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, character, person, man, woman, dog, cat, animal",
    },
    {
        "name": "Disney",
        "prompt": "{prompt}. ((A Pixar animation background)),  pixar-style, studio anime, Disney, high-quality, background, no character, only scene",
        "negative_prompt": "lowres, bad anatomy, bad hands, text, bad eyes, bad arms, bad legs, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry, grayscale, noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo, character, person, man, woman, dog, cat, animal",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed, background, no character, only scene",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, character, person, man, woman, dog, cat, animal",
    },
    {
        "name": "Comics",
        "prompt": "comic style {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed, background",
        "negative_prompt": "photograph, deformed, glitch, noisy, realistic, stock photo, character, person, man, woman, dog, cat, animal",
    },
    {
        "name": "Line",
        "prompt": "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics, background",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, character, person, man, woman, dog, cat, animal",
    }
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}