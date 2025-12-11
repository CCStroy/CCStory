from openai import OpenAI
import numpy as np
import os
import argparse
from pathlib import Path
import json

from template import story_template, scene_template, character_template, background_template

# ==========
# private key
# ==========
OPENAI_API_BASE = "https://api.openai.com/v1/" # NOTE: [User specified]
OPENAI_API_KEY = "sk-"
OPENAI_MODEL ='gpt-4' #'gpt-3.5-turbo' # 'gpt-4'

    
def parse_args():
    parser = argparse.ArgumentParser(description='Generating about story and layout')
    # parser.add_argument('--input_characters', default='a mouse, John. and a cat, Cindy', help='Enter your character name and kind')
    parser.add_argument('--input_characters', nargs='*', action=ParseKwargs) # 'Lemi':'a mouse' 'Cindy':'a cat'
    parser.add_argument('--mode', choices=['story', 'layout'], default='story', help='Generating mode - story or layout')
    parser.add_argument('--save_path', default='./output/story', help='saving path generated texts')
    
    args = parser.parse_args()
    return args

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split(':')
            getattr(namespace, self.dest)[key] = value
    
class TextGenerator():
    def __init__(self, args, api_key = OPENAI_API_KEY, model = OPENAI_MODEL):
        self.args = args
        self.api_key = api_key
        self.model = model

    def get_response(self, prompt):
        api_key=self.api_key
        model=self.model
        mode=self.args.mode
        
        client = OpenAI(
            api_key = api_key, 
        )
        
        if mode == 'story': # story_template & scene_template
            response = client.chat.completions.create(
                model = model,
                messages =[
                    {"role": "user", "content" : f"""{prompt}"""}
                ],  
            )
        elif mode == "layout": # box_template
            response = client.chat.completions.create(
                model = model,
                messages =[
                    {"role": "assistant", "content" : f"""{prompt}"""}
                ],
            )
        return response

    def get_global_story(self): # using story_template
        prompt = ""

        for key in self.args.input_characters.keys():
            txt = key +' is ' + self.args.input_characters[key] + '. '
            prompt = prompt + txt
        
        base = f"Write a short story about {prompt} \n"
        story_prompt = base + story_template
        
        if self.args.mode == 'story':
            res = self.get_response(story_prompt)
            print("Generating story")
        return res 

    def get_scene_story(self, story_res): # using scene_template
        scene_prompt = 'Story : ' + story_res + "\n" + scene_template
        if self.args.mode == 'story':
            res = self.get_response(scene_prompt)
            print("Generating scene story")
        return res
    
    def get_character_story(self, scene_res):
        characters = list(self.args.input_characters.keys())
        characters = " and ".join(characters)
        
        template = character_template.replace("{characters}", characters)
        chacter_prompt = template + "\n" + scene_res 
        if self.args.mode == 'story':
            res = self.get_response(chacter_prompt)
            print("Describing character")
        return res
    
    def get_back_scene_story(self, scene_res):
        background_prompt = background_template + "\n" + scene_res
        res = self.get_response(background_prompt)
        print("Generating backgrounds scene")
        return res
    
    def save_story(self):
        story_path = self.args.save_path + '/story.txt'
        scene_path = self.args.save_path + '/scene.txt'
        chacter_path = self.args.save_path + '/characters.txt'
        background_path = self.args.save_path +'/background.txt'
       
        story_res = self.get_global_story()
        scene_res = self.get_scene_story(story_res.choices[0].message.content)
        character_res = self.get_character_story(scene_res.choices[0].message.content)
        background_res = self.get_back_scene_story(scene_res.choices[0].message.content)
        
        with open(story_path, "w+") as file:
            file.write(story_res.choices[0].message.content)
        
        with open(scene_path, "w+") as file:
            file.write(scene_res.choices[0].message.content)
            
        with open(chacter_path, "w+") as file:
            file.write(character_res.choices[0].message.content)       
        
        with open(background_path, "w+") as file:
            file.write(background_res.choices[0].message.content)



if __name__ == "__main__":
    args = parse_args()
    Path(args.save_path).mkdir(exist_ok=True, parents=True)

    # json으로 설명이 같이 저장될 수 있도록 코드를 수정해야합니다..
    with open(args.save_path+'/characters.json', 'w') as file:
        json.dump(args.input_characters, file, indent=4)
        
    story = TextGenerator(args=args)
    story.save_story()
