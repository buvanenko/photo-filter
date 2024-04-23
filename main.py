import numpy as np
import gradio as gr
import cv2
from PIL import Image, ImageEnhance


def pencil_sketch(input_img) -> Image:
    img = input_img
    input_img = np.array(input_img)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    dst_sketch, dst_color_sketch = cv2.pencilSketch(input_img, sigma_s=50, sigma_r=0.07, shade_factor=0.08)
    bw = img.convert('L')
    bright_im = ImageEnhance.Brightness(bw).enhance(1.2)
    contrast_im = ImageEnhance.Contrast(bright_im).enhance(1.2)
    contour = Image.fromarray(dst_sketch)
    contour = contour.resize(contrast_im.size)
    result_contour = Image.blend(contrast_im, contour, 0.25)
    pencil = Image.open('pencil.png')
    pencil = pencil.resize(img.size)
    pencil = pencil.convert('RGBA')
    pencil.putalpha(int(128))
    pencil = pencil.resize(img.size)
    result = Image.alpha_composite(result_contour.convert('RGBA'), pencil)
    result = ImageEnhance.Contrast(result).enhance(1.2)
    return result


demo = gr.Interface(
    pencil_sketch, 
    gr.Image(type='pil'), 
    "image",
    title="Pencil Sketch",
    allow_flagging='never')

if __name__ == "__main__":
    demo.launch()