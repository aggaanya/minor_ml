"""
Helper Script to Generate AI Images for Dataset

Uses Stable Diffusion to generate food/grocery images.
"""

import argparse
import os
from pathlib import Path

try:
    from diffusers import StableDiffusionPipeline
    import torch
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not installed. Install with: pip install diffusers transformers accelerate")


def generate_images(prompts, output_dir, model_id="runwayml/stable-diffusion-v1-5", num_images=1):
    """
    Generate AI images using Stable Diffusion.
    
    Args:
        prompts: List of text prompts
        output_dir: Directory to save images
        model_id: HuggingFace model ID
        num_images: Number of images per prompt
    """
    if not DIFFUSERS_AVAILABLE:
        print("Error: diffusers library not available")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype
    )
    pipe = pipe.to(device)
    
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow. Consider using GPU.")
    
    # Generate images
    for prompt in prompts:
        print(f"\nGenerating: {prompt}")
        for i in range(num_images):
            try:
                image = pipe(prompt, num_inference_steps=50).images[0]
                
                # Save as JPEG (important for Q-table analysis)
                fname = f"{prompt[:30].replace(' ', '_').replace('/', '_')}_{i+1}.jpg"
                fname = "".join(c for c in fname if c.isalnum() or c in ('_', '-', '.'))
                output_path = os.path.join(output_dir, fname)
                image.save(output_path, 'JPEG', quality=85)
                print(f"  Saved: {output_path}")
            except Exception as e:
                print(f"  Error generating image: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate AI images using Stable Diffusion')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5',
                       help='Stable Diffusion model ID')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images per prompt')
    parser.add_argument('--prompts_file', type=str, default=None,
                       help='Text file with prompts (one per line)')
    parser.add_argument('--prompt', type=str, action='append', default=[],
                       help='Single prompt (can be used multiple times)')
    
    args = parser.parse_args()
    
    # Collect prompts
    prompts = []
    
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r') as f:
            prompts.extend([line.strip() for line in f if line.strip()])
    
    prompts.extend(args.prompt)
    
    if not prompts:
        # Default prompts for food/grocery
        prompts = [
            "a photo of fresh apples on a wooden table",
            "grocery store shelf with packaged food products",
            "close-up of a burger with fries on a plate",
            "colorful vegetables arranged on a white background",
            "bottles of juice and drinks on a shelf",
            "fresh bread and pastries in a bakery",
            "packaged snacks and chips on display",
            "fresh fish and seafood on ice",
            "organic fruits and vegetables in a basket",
            "canned goods and pantry items on shelves"
        ]
        print("Using default prompts. Provide --prompts_file or --prompt for custom prompts.")
    
    print(f"Generating {len(prompts)} prompts, {args.num_images} images each...")
    generate_images(prompts, args.output_dir, args.model, args.num_images)
    print(f"\nDone! Images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
