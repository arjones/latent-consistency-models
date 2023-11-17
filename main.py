import argparse
import os
import time
import json

from copy import deepcopy


from diffusers import DiffusionPipeline
import torch

import warnings
warnings.filterwarnings("ignore")

class Predictor:
    def __init__(self):
        self.pipe = self._load_model()

    def _load_model(self):
        model = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", 
                                          local_files_only=True)

        # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        model.to(torch_device="mps", torch_dtype=torch.float32).to('mps:0')
        model.enable_attention_slicing()
        return model
    
    def predict(self, prompt: str, negative_prompt: str, seed: int):
        seed = seed or int.from_bytes(os.urandom(2), "big")
        torch.manual_seed(seed)

        # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
        num_inference_steps = 4

        images = self.pipe(prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=640,
                        height=320,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=8.0,
                        lcm_origin_steps=50,
                        num_images_per_prompt=4,
                        output_type="pil").images
        
        return {
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "images": images
        }
    
    def save(self, prediction):
      timestamp = time.strftime("%Y%m%d-%H%M%S")
      output_dir = f'output/{timestamp}-{prediction["seed"]}'
      print(f'\n{output_dir}\n\n')
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)

      # extract and delete images from dict
      images = deepcopy(prediction["images"])
      del prediction["images"]

      with open(f'{output_dir}/index.json', 'w') as json_file:
          json.dump(prediction, json_file, indent=4)

      for idx, image in enumerate(images):
        filename = f'{output_dir}/img_{idx}.png'
        image.save(filename)


def prompt_match(prompt, neg_prompt):
    """Match positive and negative prompt"""
    neg_prompt = neg_prompt or []
    neg_ext = neg_prompt + [None] * (len(prompt) - len(neg_prompt))
    return zip(prompt, neg_ext)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images based on text prompts.")
    parser.add_argument("-p", "--prompt", action='append', help="A text prompt", required=True)
    parser.add_argument("-np", "--negprompt", action='append', help="A negative prompt")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation.")
    return parser.parse_args()

def main():
    args = parse_args()
    predictor = Predictor()

    prompts = prompt_match(args.prompt, args.negprompt)
    for p, np in prompts:
        print(f'Processing: (p={p}, np={np}, seed={args.seed})\n\n')
        prediction = predictor.predict(prompt = p,
                                      negative_prompt = np,
                                      seed=args.seed)

        predictor.save(prediction=prediction)

if __name__ == "__main__":
    main()
