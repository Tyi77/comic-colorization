import argparse
import gc
import logging
import os
import time
import cv2

import gradio as gr
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from rich.traceback import install
from skimage.color import lab2rgb, rgb2lab
from skimage.util import img_as_ubyte
from torchvision import transforms

from src.conditional.conditional_autoencoder import ConditionalAutoencoder
from src.shading.shader import Shader
from src.utils import resize_max


def resize_to_nearest(img: Image.Image):
    img = resize_max(img, conf.params.max_size)
    w, h = img.size
    img = img.resize((((w + 7) // 8) * 8, ((h + 7) // 8) * 8), Image.Resampling.LANCZOS)
    return img


def blend_lch(
    rgb_a: Image.Image, rgb_b: Image.Image, lratio=0.0, cratio=0.0, hratio=0.0
):
    lch_a = rgb2lab(np.array(rgb_a, dtype=np.float64) / 255.0)
    lch_b = rgb2lab(np.array(rgb_b, dtype=np.float64) / 255.0)

    l_c = lch_a[..., 0] * (1.0 - lratio) + lch_b[..., 0] * lratio
    c_c = lch_a[..., 1] * (1.0 - cratio) + lch_b[..., 1] * cratio
    h_c = lch_a[..., 2] * (1.0 - hratio) + lch_b[..., 2] * hratio
    lch_c = np.stack((l_c, c_c, h_c), axis=-1)

    rgb_c = Image.fromarray(img_as_ubyte(lab2rgb(lch_c)))
    return rgb_c


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_path",
        type=str,
        required=False,
        help="Path to the configuration file",
        default="./app.yaml",
    )
    args = parser.parse_args()
    return args


logging.basicConfig(level=logging.INFO)
install(show_locals=False)
args = parse_args()

logging.info("Loading configuration")
conf = OmegaConf.load(args.conf_path)

logging.info("Setting up models")
with torch.no_grad():
    cond_autoencoder = ConditionalAutoencoder.load_from_saved(
        conf.paths.pretrained_ckpt,
        conf.paths.pretrained_yaml,
        conf.paths.conditional_ckpt,
    ).to(device="cuda", dtype=torch.bfloat16)
    cond_autoencoder.eval()

    shader = Shader(device="cuda", generator_path=conf.paths.shader_ckpt)

with gr.Blocks(theme=gr.themes.Monochrome(), title="inkn'hue") as demo:
    gr.Markdown(
        "### inkn'hue: Aligning Style2Paints for Faithful Controllable Manga Colorization"
    )
    with gr.Tab("Source") as src:
        with gr.Group():
            with gr.Row():
                src_sketch = gr.Image(
                    label="sketch",
                    height=700,
                    type="pil",
                    image_mode="RGB",
                )
                src_s2p = gr.Image(
                    label="style2paints",
                    height=700,
                    type="pil",
                    image_mode="RGB",
                )
            runDataset_btn = gr.Button("Run Dataset")
    with gr.Tab("Shade & Colorize") as gen:
        with gr.Group():
            gen_img = gr.Image(
                label="generated",
                height=650,
                type="pil",
                image_mode="RGB",
                interactive=False,
            )
            gen_btn = gr.Button("Generate")
    with gr.Tab("Postprocess") as post:
        with gr.Group():
            post_img = gr.Image(
                label="generated postprocessed",
                height=650,
                type="pil",
                image_mode="RGB",
                interactive=False,
            )
            post_cratio = gr.Slider(
                label="Color ratio (0 = generated, 1 = style2paints)",
                maximum=1,
                step=1e-04,
                value=0.8,
            )
    with gr.Tab("Export") as exp:
        with gr.Group():
            exp_img = gr.Image(
                label="results",
                height=650,
                type="pil",
                image_mode="RGB",
                interactive=False,
            )
            with gr.Row():
                exp_dir = gr.Textbox(
                    "./results", label="Export directory", max_lines=1, interactive=True
                )
                exp_name = gr.Textbox(
                    "",
                    label="Export filename",
                    max_lines=1,
                    interactive=True,
                    placeholder="<datetime>.png",
                )
            exp_btn = gr.Button("Save")


    @runDataset_btn.click()
    def run_dataset():
        folder1 = './append/Sketch'
        folder2 = './append/Style2Paints'
        # 獲取兩個資料夾中的檔案列表
        files1 = sorted(os.listdir(folder1))  # 排序以確保順序一致
        files2 = sorted(os.listdir(folder2))
        # 同時遍歷兩個資料夾中的檔案
        for file1, file2 in zip(files1, files2):
            path1 = os.path.join(folder1, file1)
            path2 = os.path.join(folder2, file2)
            # 確認是檔案而非資料夾
            if os.path.isfile(path1) and os.path.isfile(path2):
                print(f"Processing files: {path1} and {path2}")
                img1 = Image.open(path1).convert("RGB")
                img2 = Image.open(path2).convert("RGB")
                gen = generate(img1, img2)
                print(type(gen))
                exp_img = postprocess(gen, img2, 0.8)
                exp_path = os.path.join('./append/Result', file1)
                exp_img.save(exp_path)

    @gen_btn.click(inputs=[src_sketch, src_s2p], outputs=gen_img)
    @torch.no_grad()
    def generate(sketch: Image.Image, s2p: Image.Image):
        def to_tensor(img: Image.Image):
            pil_to_tensor = transforms.PILToTensor()
            img = pil_to_tensor(img)
            img = ((img / 255.0) * 2.0 - 1.0).clamp(-1.0, 1.0)
            img = img.to(device="cuda", dtype=torch.bfloat16)
            img = rearrange(img, "c h w -> 1 c h w")
            return img
        
        if sketch is None or s2p is None:
            return gr.Image(interactive=False)
        print(sketch.size)
        print(s2p.size)
        shaded = shader.shade(np.array(sketch), conf.params.max_size)
        shaded = to_tensor(resize_to_nearest(shaded))
        s2p = to_tensor(resize_to_nearest(s2p))

        z = cond_autoencoder.encode(s2p)
        conds_z = cond_autoencoder.encode_cond(shaded)
        print("z.sample() shape:", z.sample().shape)
        print("conds_z shape:", [c.shape for c in conds_z])
        gc.collect()
        torch.cuda.empty_cache()
        y = cond_autoencoder.decode(z.sample(), conds_z)

        result = ((y[0].detach() + 1.0) * 0.5 * 255.0).clamp(0, 255)
        result = rearrange(result, "c h w -> h w c")
        result = result.to(device="cpu", dtype=torch.uint8).numpy()
        result = resize_to_nearest(Image.fromarray(result))

        gc.collect()
        torch.cuda.empty_cache()

        # return gr.Image(result, interactive=True)
        return result

    @gen_img.change(inputs=[gen_img, src_s2p, post_cratio], outputs=[post_img, exp_img])
    @src_s2p.change(inputs=[gen_img, src_s2p, post_cratio], outputs=[post_img, exp_img])
    @post_cratio.change(
        inputs=[gen_img, src_s2p, post_cratio], outputs=[post_img, exp_img]
    )
    def postprocess(gen: Image.Image, s2p: Image.Image, cratio):
        if gen is None or s2p is None:
            ret = gr.Image(interactive=False)
            return [ret, ret]

        s2p = resize_to_nearest(s2p)
        print(type(gen))
        print(type(s2p))
        combined = blend_lch(gen, s2p, 0, cratio, cratio)

        ret = gr.Image(combined, interactive=True)
        # return [ret, ret]
        return combined

    @exp_btn.click(inputs=[exp_dir, exp_name, exp_img])
    def save_results(dir, name, img: Image.Image):
        os.makedirs(dir, exist_ok=True)
        if name == "":
            name = f"{time.strftime('%Y%m%d_%H%M%S')}.png"
        save_path = f"{dir}/{name}"
        img.save(save_path)
        gr.Info(f"Saved to {save_path}")


demo.queue().launch(show_error=True)
