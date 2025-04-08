from typing import List, Tuple, Union
import pathlib
from pathlib import Path
import tensorflow as tf
import re
from PIL import Image, UnidentifiedImageError, ImageFont, ImageDraw
import numpy as np
import skimage.color
import skimage.filters
import skimage.measure
import math
from Core.HelperFunctions import printRep
from typing import Tuple
import cv2
import os
import matplotlib.pyplot as plt
import io
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

def ComputeOutline(image: np.ndarray):
    return skimage.filters.sobel(image) > 0.00001
# Find the best focus organoid using laplacian
def laplacian_variance(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return cv2.Laplacian(gray, cv2.CV_64F).var()

def ExtractOrganoidRegion(image, bbox):
    minr, minc, maxr, maxc = bbox
    return image[minr:maxr, minc:maxc]

def ImagesToHeatmaps(images: np.ndarray):
    print("Preparing heatmaps...", end="", flush=True)
    heatmaps = np.zeros(list(images.shape) + [3], dtype=np.uint8)
    for i in range(images.shape[0]):
        printRep(str(i) + "/" + str(images.shape[0]))
        image = images[i]
        minimum = np.min(image)
        maximum = np.max(image)
        hue = 44.8 / 360
        h = np.ones_like(image) * hue
        s = np.minimum(1, 2 - 2 * (image - minimum) / (maximum - minimum))
        v = np.minimum(1, 2 * (image - minimum) / (maximum - minimum))
        concat = np.stack([h, s, v], -1)
        converted = skimage.color.hsv2rgb(concat)
        heatmaps[i] = (converted * 255)
    printRep(None)
    print("Done.")
    return heatmaps


def LabeledImagesToColoredImages(images: np.ndarray, colors=None, fontSize=0):
    if colors is None:
        colors = [(255, 0, 0),
                  (0, 255, 0),
                  (0, 0, 255),
                  (255, 255, 0),
                  (255, 0, 255),
                  (0, 255, 255),
                  (128, 128, 128)]
    cycles = math.ceil(float(np.max(images)) / len(colors))
    colorMap = np.asarray([(0, 0, 0)] + colors * cycles, dtype=np.uint8)
    colorized = colorMap[images]

    if fontSize > 0:
        font = ImageFont.load_default()
        #font = ImageFont.truetype("arial.ttf", fontSize)
        for i in range(images.shape[0]):
            image = Image.fromarray(colorized[i])
            drawer = ImageDraw.Draw(image)
            for rp in skimage.measure.regionprops(images[i]):
                x, y = reversed(rp.centroid)
                drawer.text((x, y), str(rp.label), anchor="ms", fill=(255, 255, 255), font=font)
            colorized[i] = np.asarray(image)
    return colorized


def NumFrames(image: Image):
    return getattr(image, "n_frames", 1)


def GetFrames(image: Image.Image):
    for i in range(NumFrames(image)):
        image.seek(i)
        yield image
    image.seek(0)


def PILImageForFrameInList(i, images: List[Image.Image]):
    for image in images:
        if i < NumFrames(image):
            return image
        i -= NumFrames(image)


def ConvertImagesToStacks(images: np.ndarray, originalImages: List[Image.Image]):
    stacks = []
    i = 0
    for originalImage in originalImages:
        start = i
        end = i + NumFrames(originalImage)
        stacks.append(images[start:end])
        i = end
    return stacks


def ConvertImagesToPILImageStacks(images: np.ndarray, originalImages: List[Image.Image],
                                  resize=True):
    stacks = ConvertImagesToStacks(images, originalImages)
    if resize:
        return [
            [Image.fromarray(d).resize(o.size, resample=Image.Resampling.NEAREST) for d in stack]
            for o, stack in
            zip(originalImages, stacks)]
    else:
        return [[Image.fromarray(d) for d in stack] for stack in stacks]


def SavePILImageStack(stack: List[Image.Image], path: pathlib.Path):
    if stack[0].mode[0] == "I":
        path = path.parent / (path.stem + ".tif")

    #Process RGBA mode, JPEG does not support transparent channels
    if stack[0].mode == "RGBA":
        stack = [img.convert("RGB") for img in stack]
    
    if len(stack) == 1:
        stack[0].save(path)
    else:
        stack[0].save(path, save_all=True, append_images=stack[1:], compression=None)


def SaveAsGIF(images: np.ndarray, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(images[0]).convert(mode="RGB").save(path, save_all=True,
                                                        append_images=[
                                                            Image.fromarray(im).convert(mode="RGB")
                                                            for im in images[1:]],
                                                        loop=0)


def LoadPILImages(source: Union[pathlib.Path, List[pathlib.Path]]) -> List[Image.Image]:
    def OpenAndSkipErrors(path: List[pathlib.Path]):
        for p in path:
            try:
                i = Image.open(p)
                yield i
            except UnidentifiedImageError:
                pass

    if isinstance(source, list):
        return sum([LoadPILImages(i) for i in source], [])
    if source.is_dir():
        # Load directory
        matches = sort_paths_nicely([path for path in source.iterdir() if path.is_file()])
        if len(matches) == 0:
            raise Exception(
                "Could not find any images in directory '" + str(source.absolute()) + "'.")
        return list(OpenAndSkipErrors(matches))
    if source.is_file():
        return [Image.open(source)]

    # Handle regular expression paths
    matches = sort_paths_nicely(
        [path for path in source.parent.glob(source.name) if path.is_file()])
    if len(matches) == 0:
        raise Exception("Could not find any images matching '" + str(source.absolute()) + "'.")
    return list(OpenAndSkipErrors(matches))

def DrawRegionsOnImagesOriginal(labeledImages: np.ndarray, images: np.ndarray,
                        textColor: Tuple[int, int, int],
                        fontSize: int, overlayColor: Tuple[int, int, int]):
    #font = ImageFont.truetype("arial.ttf", fontSize)
    images = np.repeat(images[:, :, :, None], 3, axis=-1)
    outlined = np.zeros(images.shape[:-1], dtype=bool)
    for i in range(images.shape[0]):
        outlined[i] = ComputeOutline(labeledImages[i])
    drawnImages = np.where(outlined[:, :, :, None], overlayColor, images).astype(np.uint8)
    for i in range(images.shape[0]):
        image = Image.fromarray(drawnImages[i])
        drawer = ImageDraw.Draw(image)
        for rp in skimage.measure.regionprops(labeledImages[i]):
            x, y = reversed(rp.centroid)
            #drawer.text((x, y), str(rp.label), anchor="ms", fill=textColor, font=font)
        drawnImages[i] = np.asarray(image)
    return drawnImages

def DrawRegionsOnImages(labeledImages: np.ndarray, images: np.ndarray,
                        textColor: Tuple[int, int, int], fontSize: int,
                        overlayColor: Tuple[int, int, int]) -> np.ndarray:
    print("Function DrawRegionsOnImages called.")
    save_folder = pathlib.Path("./2")
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"Save folder '{save_folder}' created or already exists.")
    font = ImageFont.load_default()
    images = np.repeat(images[:, :, :, None], 3, axis=-1)
    #outlined = np.zeros(images.shape[:-1], dtype=bool)
    
    #for i in range(images.shape[0]):
        #outlined[i] = ComputeOutline(labeledImages[i])
    
    #drawnImages = np.where(outlined[:, :, :, None], overlayColor, images).astype(np.uint8)
    drawnImages = images.astype(np.uint8)
    
    for i in range(images.shape[0]):
        image = Image.fromarray(drawnImages[i])
        drawer = ImageDraw.Draw(image)
        rp_idx = 0
        for rp in skimage.measure.regionprops(labeledImages[i]):

            y, x = rp.centroid  


            radius = max(rp.major_axis_length, rp.equivalent_diameter) / 2 + 1


            min_row = max(int(x - radius), 0)
            min_col = max(int(y - radius), 0)
            max_row = min(int(x + radius), image.size[0])
            max_col = min(int(y + radius), image.size[1])
            cropped_image = image.crop((min_row, min_col, max_row, max_col))
            

            #print(f"Saving cropped image {i}_{rp_idx} to {save_folder}")

            crop_path = save_folder / f"image_{i}_region_{rp_idx}.png"
            cropped_image.save(crop_path)

            drawer.rectangle([min_row, min_col, max_row, max_col], outline="red", width=2)
            rp_idx += 1

            #drawer.text((x, y), str(rp.label), anchor="ms", fill=textColor, font=font)
        
        drawnImages[i] = np.asarray(image)
    
    return drawnImages

def DrawRegionsOnImagesWithText(model: tf.keras.layers.Layer, labeledImages: np.ndarray, images: np.ndarray,
                        textColor: Tuple[int, int, int], fontSize: int,
                        overlayColor: Tuple[int, int, int]) -> np.ndarray:
    print("Function DrawRegionsOnImages called.")
    save_folder = pathlib.Path("./2")
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"Save folder '{save_folder}' created or already exists.")
    font = ImageFont.load_default()
    images = np.repeat(images[:, :, :, None], 3, axis=-1)
    #outlined = np.zeros(images.shape[:-1], dtype=bool)
    
    #for i in range(images.shape[0]):
        #outlined[i] = ComputeOutline(labeledImages[i])
    
    #drawnImages = np.where(outlined[:, :, :, None], overlayColor, images).astype(np.uint8)
    drawnImages = images.astype(np.uint8)
    
    for i in range(images.shape[0]):
        image = Image.fromarray(drawnImages[i])
        drawer = ImageDraw.Draw(image)
        rp_idx = 0
        for rp in skimage.measure.regionprops(labeledImages[i]):

            y, x = rp.centroid 


            radius = max(rp.major_axis_length, rp.equivalent_diameter) / 2 + 1


            min_row = max(int(x - radius), 0)
            min_col = max(int(y - radius), 0)
            max_row = min(int(x + radius), image.size[0])
            max_col = min(int(y + radius), image.size[1])
            cropped_image = image.crop((min_row, min_col, max_row, max_col))
            

            #print(f"Saving cropped image {i}_{rp_idx} to {save_folder}")

            crop_path = save_folder / f"image_{i}_region_{rp_idx}.png"
            cropped_image.save(crop_path)
            prediction_text = str(rp_idx)
            if model:
                frame = np.array(cropped_image)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame = cv2.resize(frame, (224, 224))
                frame_array = np.array(frame, dtype=np.float32)
                frame_array = preprocess_input(frame_array)
                input_data = np.expand_dims(frame_array, axis=0)
                prediction = model(input_data)
                predicted_tensor = prediction['dense_5']
                val = predicted_tensor.numpy().item()
                prediction_text = "T" if val >= 0.5 else "S"

            drawer.rectangle([min_row, min_col, max_row, max_col], outline="red", width=2)


            drawer.text((x, y), prediction_text, anchor="ms", fill=textColor, font=font)
            rp_idx += 1
        
        drawnImages[i] = np.asarray(image)
    
    return drawnImages

def ComputeOutline(labelImage):

    return labelImage > 0

def DrawRegionsAndSizeDistributions(labeledImages: np.ndarray, images: np.ndarray,
                                    textColor: Tuple[int, int, int], fontSize: int,
                                    overlayColor: Tuple[int, int, int]) -> np.ndarray:
    images = np.repeat(images[:, :, :, None], 3, axis=-1)
    outlined = np.zeros(images.shape[:-1], dtype=bool)
    distribution_images_np = []
    pixel_to_um = 2 

    for i in range(images.shape[0]):
        outlined[i] = ComputeOutline(labeledImages[i])
    
    drawnImages = np.where(outlined[:, :, :, None], overlayColor, images).astype(np.uint8)

    for i in range(labeledImages.shape[0]):
        radii_um = []
        for rp in skimage.measure.regionprops(labeledImages[i]):

            radius_um = ((max(rp.major_axis_length, rp.equivalent_diameter) / 2) * pixel_to_um * 4)
            radii_um.append(radius_um)

        plt.figure(figsize=(5, 5))  
        plt.hist(radii_um, bins=20, color='skyblue')
        plt.title('Size Distribution')
        plt.xlabel('Radius (um)')
        plt.ylabel('Frequency')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close() 
        buf.seek(0)
        distribution_image_pil = Image.open(buf)
        

        distribution_image_resized = np.array(distribution_image_pil.resize((images.shape[2], images.shape[1])))
        distribution_images_np.append(distribution_image_resized)

    distribution_images_np = np.array(distribution_images_np)

    return distribution_images_np

def sort_paths_nicely(paths: List[pathlib.Path]):
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(paths, key=lambda x: alphanum_key(x.name))
