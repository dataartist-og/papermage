from html import entities
from papermage.predictors.base_predictors.base_predictor import BasePredictor
from papermage.magelib import (
    Document,
    Entity,
    Box,
    Span,
    Metadata,
    ImagesFieldName,
    PagesFieldName
)
from typing import List
import os
from tqdm.auto import tqdm
import cv2
import json
import yaml
import time
import pytz
import datetime
import argparse
import shutil
import base64
import torch
import numpy as np
import requests
from litellm import completion
import litellm
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.nap import sleep
from litellm.caching import Cache
litellm.cache = Cache(type="disk")

openai.api_key = os.getenv("OPENAI_API_KEY")
# litellm.set_verbose = True

from PIL import ImageChops, Image, ImageDraw, ImageFont
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor
# import sys; sys.path.append('/content/PDF-Extract-Kit')
from papermage.modules.latex2png import tex2pil, zhtext2pil
from papermage.modules.extract_pdf import load_pdf_fitz
from papermage.modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from papermage.modules.self_modify import ModifiedPaddleOCR
from papermage.modules.post_process import get_croped_image, latex_rm_whitespace
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
import torchvision.transforms.functional as tvt

LATEX_STR_VALIDATION_PROMPT = "You are a LaTeX expert. Your job is to correct malformed latex. Only output the correct latex, even if the original is correct. Don't add any comments, just output the required latex string. Your response will directly be passed to the renderer, so if you add anything else you will fail.  Do not provide the open/close latex tags like \\(\\) \\[\\] or $,$$. Give the correct latex, nothing else."
LATEX_STR_VALIDATION_W_IMG_PROMPT = f"You are a LaTeX expert. Your job is to validate and correct (potentially malformed) latex, given an image of what the rendered latex SHOULD look like.You will be provided an image, and a latex string. Your job is to check whether the string will match the provided image when rendered, and if it doesn't, provide a corrected latex string. Only output the correct latex, even if the original is correct. Don't add any comments, just output the required latex string. Your response will directly be passed to the renderer, so if you add anything else you will fail. Do not provide the open/close latex tags like \\(\\) \\[\\] or $,$$. Give the correct latex, nothing else."

def mfd_model_init(weight):
    mfd_model = YOLO(weight)
    return mfd_model

def latex_to_image(latex_str, dpi=300):
    """
    Render a LaTeX string to a PNG image and return as a PIL Image object.

    Args:
        latex_str (str): The LaTeX string to render.
        dpi (int): Dots per inch (resolution) of the output image.

    Returns:
        PIL.Image.Image: The rendered image as a PIL Image object.
    """
    # Create a figure and axis with no frame or axis
    fig, ax = plt.subplots(figsize=(0.01, 0.01))
    ax.text(0.5, 0.5, f"${latex_str}$", fontsize=20, ha='center', va='center')
    ax.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the figure to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Load the image from the buffer
    buf.seek(0)
    image = Image.open(buf)
    
    return image


def mfr_model_init(weight_dir, device='cpu'):
    args = argparse.Namespace(cfg_path="modules/UniMERNet/configs/demo.yaml", options=None)
    cfg = Config(args)
    cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(device)
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    return model, vis_processor

def pil_image_to_base64(image, format="PNG"):
    """
    Encode a PIL Image to a base64 string.

    Args:
        image (PIL.Image.Image): The PIL image to encode.
        format (str): The format to use for the image (e.g., "PNG", "JPEG").

    Returns:
        str: The base64 encoded string of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def validate_and_correct_latex(latex, image):
    # First, try to render the LaTeX
    if isinstance(image, str):
        original_image = Image.open(image)
    elif isinstance(image, Image.Image):
        original_image = image
    encoded_image = pil_image_to_base64(original_image)
    try:
        rendered_image = latex_to_image(latex)
        
        # Compare the rendered image with the original image
        
        # Resize the rendered image to match the original image size
        rendered_image = rendered_image.resize(original_image.size)
        
        # Calculate the difference between the images
        diff = ImageChops.difference(original_image, rendered_image)
        
        # If the difference is significant, consider the LaTeX as potentially invalid
        if diff.getbbox() is not None and (diff.getbbox()[2] - diff.getbbox()[0]) * (diff.getbbox()[3] - diff.getbbox()[1]) > 0.1 * original_image.size[0] * original_image.size[1]:
            raise ValueError("Rendered image differs significantly from the original")
        
        return latex  # If no exception and images are similar, LaTeX is likely valid
    except:
        # If LaTeX is invalid or renders incorrectly, try to correct it using Mathpix API
        try:
            raise ValueError("LaTeX is invalid or renders incorrectly")
            response = requests.post(
                "https://api.mathpix.com/v3/text",
                json={"src": latex, "formats": ["latex_simplified"]},
                headers={
                    "app_id": "YOUR_MATHPIX_APP_ID",
                    "app_key": "YOUR_MATHPIX_APP_KEY",
                }
            )
            corrected_latex = response.json()["latex_simplified"]
            return corrected_latex
        except:
            # If Mathpix fails, use GPT-4 as a fallback
            try:                
                messages = [
                    {"role": "system", "content": LATEX_STR_VALIDATION_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": latex},
                    ]}
                ]
                
                response = completion(
                    model="gpt-4o-mini",
                    messages=messages
                )
                
                corrected_latex = response.choices[0].message.content
                print ("GPT-4o-mini response: ", corrected_latex)
                try:
                    latex_to_image(corrected_latex)
                    return corrected_latex
                except:
                    raise ValueError("GPT-4o-mini response is invalid")
            except:
                # If gpt-4o-mini fails, use gpt-4o with image
                try:
                    messages = [
                        {"role": "system", "content": LATEX_STR_VALIDATION_W_IMG_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": latex},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                        ]}
                    ]
                    
                    response = completion(
                        model="gpt-4o",
                        messages=messages
                    )
                    
                    corrected_latex = response.choices[0].message.content
                    try:
                        latex_to_image(corrected_latex)
                        return corrected_latex
                    except:
                        raise ValueError("GPT-4o response is invalid")

                except:
                    # If all else fails, return the original LaTeX
                    return latex


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
        return image


class MathPredictor(BasePredictor):
    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [PagesFieldName, ImagesFieldName]

    def __init__(self, config_path='configs/model_configs.yaml', verbose=False):
        # Initialize YOLO model using the implementation from PDF-Extract-Kit
        with open(config_path) as f:
            model_configs = yaml.load(f, Loader=yaml.FullLoader)
        self.img_size = model_configs['model_args']['img_size']
        self.conf_thres = model_configs['model_args']['conf_thres']
        self.iou_thres = model_configs['model_args']['iou_thres']
        self.device = model_configs['model_args']['device']
        self.dpi = model_configs['model_args']['pdf_dpi']
        self.mfd_model = mfd_model_init(model_configs['model_args']['mfd_weight'])
        self.mfr_model, mfr_vis_processors = mfr_model_init(model_configs['model_args']['mfr_weight'], device=self.device)
        self.mfr_transform = transforms.Compose([mfr_vis_processors, ])
        self.verbose = verbose

    def _predict(self, doc: Document) -> List[Entity]:
        """Returns a list of Entities for the detected layouts for all pages

        Args:
            document (Document):
                The input document object

        Returns:
            List[Entity]:
                The returned Entities for the detected layouts for all pages
        """
        document_prediction = []

        images = doc.get_layer(name=ImagesFieldName)
        resize_transform = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize to fixed size that is divisible by 32
            # transforms.ToTensor(),          # Convert to tensor with shape (C, H, W)
        ])

        for image_index, pm_image in enumerate(tqdm(images)):
            # Convert to tensor and resize
            image = pm_image.to_array()
            print (image_index, image.shape)

            img_H, img_W = image.shape[0], image.shape[1]
            #image = resize_transform(image)
            
            # Add batch dimension
            #image = image.unsqueeze(0)  # Now shape is (1, C, H, W)
            
            # Ensure tensor is on the correct device (if using GPU)
            # if torch.cuda.is_available():
            #     image = image.to('cuda')
            
            mfd_res = self.mfd_model.predict(image, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres, verbose=self.verbose)[0]
            
            mf_image_list = []
            entities = []
            
            for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                bbox_img = get_croped_image(Image.fromarray(image), [xmin, ymin, xmax, ymax])
                mf_image_list.append(bbox_img)
                
                entity = Entity(
                    boxes=[
                        Box.from_xy_coordinates(
                            x1=xmin,
                            y1=ymin,
                            x2=xmax,
                            y2=ymax,
                            page=image_index,
                            page_width=img_W,
                            page_height=img_H
                        ).to_relative(page_width=img_W, page_height=img_H)
                    ],
                    images=[bbox_img],
                    metadata=Metadata(**{"confidence": conf.item(), "class": 'isolated_latex' if int(cla.item())==1 else 'inline_latex'})
                )
                entities.append(entity)

            # Formula recognition
            dataset = MathDataset(mf_image_list, transform=self.mfr_transform)
            dataloader = DataLoader(dataset, batch_size=128, num_workers=32)
            mfr_res = []
            for imgs in dataloader:
                imgs = imgs.to(self.device)
                output = self.mfr_model.generate({'image': imgs})
                mfr_res.extend(output['pred_str'])
            
            for entity, latex, img in zip(entities, mfr_res, mf_image_list):
                latex = latex_rm_whitespace(latex)
                corrected_latex = latex#validate_and_correct_latex(latex, img)
                entity.metadata.latex = corrected_latex

            document_prediction.extend(entities)

        return document_prediction
