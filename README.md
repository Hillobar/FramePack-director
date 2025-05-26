# FramePack-director

This is an attempt to add some quicker control over the video generation process.

New Features over the original FP-F1:
- Preview mode to get a quick idea of what the video will look like
- Generate segments one at a time, each with it's own prompt/lora
- Continue to try different generations for each new segment until you're satisfied before rendering.
- Pick any resolution
- Fixed some bugs, including the flickering between segments
- Automatically reserves the appropriate amount of memory for latents and tensor activation
- Tightened up the model loading for the experience to be faster.

Warning:
- I'm using 32G VRAM. I'm guessing it will work with less?

Usage:
- Edit demo_gradio_f1.py, line 59 lora_path to point to your lora directory.
- Add an image, or leave the image blank for t2v.
- Add a prompt in the first text box. Place a check to the left of it to indicate that the prompt should be processed. Select Quick Preview and unselect Decode Latent if you're only interested in sampling at this random number. 
- If you like the results, either deselect Quick Preview and process the full latent, or if you didn't use Quick Preview unselect this prompt and move onto the next prompt.
- Do the same thisng for the next Prompt.
- When you're happy with your segments, select Decode Latents. It will render all of the latents that are below the prompts.

Tips:
- Adjust the segment times on the right of the prompt.
- Maximum segment length in the additional parameter control the max latnet size that gets denoised. Original Framepack is ~ 1.2 s. This directly afect VRAM usage, aling with video size (resolution).
- Probably lots of bugs. It'll slowly get better over time. However, it's really useful for quickly creating good videos with expected results.

