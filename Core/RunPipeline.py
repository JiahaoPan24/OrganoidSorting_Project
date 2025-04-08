from typing import List, Optional
from pathlib import Path
from Core.Model import LoadFullModel, Detect, LoadLiteModel, PrepareImagesForModel
from Core.Identification import Cleanup, SeparateContours, DetectEdges, Label, LabelAndBoundingBox
from Core.ImageHandling import LoadPILImages, ImagesToHeatmaps, \
    LabeledImagesToColoredImages, DrawRegionsOnImagesOriginal, DrawRegionsOnImages, DrawRegionsOnImagesWithText, ConvertImagesToStacks, SaveAsGIF, DrawRegionsAndSizeDistributions
from Core.Tracking import Track, Inverse, Overlap
from PIL import Image, ImageDraw
import numpy as np
import skimage.measure
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

def draw_bounding_boxes(original_image: np.ndarray, bounding_box, padding: int) -> np.ndarray:
    # Convert the original image to a PIL image for drawing
    pil_image = Image.fromarray(original_image)
    draw = ImageDraw.Draw(pil_image)

    # Calculate the bounding box with padding
    min_row, min_col, max_row, max_col = bounding_box
    min_row, min_col = max(min_row - padding, 0), max(min_col - padding, 0)
    max_row, max_col = min(max_row + padding, original_image.shape[0]), min(max_col + padding, original_image.shape[1])

    # Draw the bounding box
    draw.rectangle([min_col, min_row, max_col, max_row], outline="red", width=2)

    return np.array(pil_image)


    
def SaveImages(data, suffix, pilImages, outputPath):
    from Core.ImageHandling import ConvertImagesToPILImageStacks, SavePILImageStack
    stacks = ConvertImagesToPILImageStacks(data, pilImages)

    for stack, pilImage in zip(stacks, pilImages):
        p = Path(pilImage.filename)
        SavePILImageStack(stack, outputPath / (p.stem + suffix + p.suffix))


def MakeDirectory(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise Exception("Could not find or create directory '" + str(path.absolute()) + "'.")


def LoadModel(modelPath: Path):
    print("Loading model...")
    print("-" * 100)
    if modelPath.is_file():
        model = LoadLiteModel(modelPath)
    else:
        model = LoadFullModel(modelPath)
    print("-" * 100)
    print("Model loaded.")
    return model


def RunPipeline(run_class: bool, modelPath: Path, imagePaths: List[Path], outputPath: Optional[Path], class_path: Path,
                threshold: float, batchSize: int, edgeSigma: float, edgeMin: float,
                edgeMax: float, minimumArea: int, fillHoles: bool, removeBorder: bool,
                detectionOutput: bool, binaryOutput: bool, separateContours: bool,
                edges: bool, colorLabeledOutput: bool, idLabeledOutput: bool,
                track: bool, overlay: bool, gif: bool, batch: bool, computeProps: bool):
    model = LoadModel(modelPath)
    # Load the images
    pilImages = LoadPILImages(imagePaths)
    preparedImages = PrepareImagesForModel(pilImages, model)
    detectionImages = Detect(model, preparedImages, batchSize)


    outputImages = {'Prepared Input': preparedImages}
    boundingBoxes = []

    def Output(name: str, data):
        if outputPath is not None:
            MakeDirectory(outputPath)
            SaveImages(data, "_" + name.lower(), pilImages, outputPath)
            if gif:
                outputImages[name] = data
        else:
            outputImages[name] = data


    if separateContours:
        edgeImages = DetectEdges(detectionImages, edgeSigma, edgeMin, edgeMax, threshold)
        if edges:
            Output('Edges', edgeImages)
        labeledImages = SeparateContours(detectionImages, edgeImages, threshold, edgeSigma)
    else:
        labeledImages = Label(detectionImages, threshold)
        #labeledImages, boundingBoxes = LabelAndBoundingBox(detectionImages, threshold, 2)
      

    cleanedImages = Cleanup(labeledImages, minimumArea, removeBorder, True)
    #print(cleanedImages)
    
    if detectionOutput:
        overlayImages = DrawRegionsOnImagesOriginal(cleanedImages, preparedImages, (255, 255, 255), 16,
                                            (0, 255, 0))
        Output('Detection', overlayImages)

    if binaryOutput:
        overlayImages = DrawRegionsAndSizeDistributions(cleanedImages, preparedImages, (255, 255, 255), 16,
                                            (0, 255, 0))
        Output('Size Distribution', overlayImages)

    if track:
        i = 0
        stacks = ConvertImagesToStacks(cleanedImages, pilImages) if batch else [cleanedImages]
        for stack in stacks:
            stack = Track(stack, 1, Inverse(Overlap))
            cleanedImages[i:(i + stack.shape[0])] = stack
            i += stack.shape[0]
        overlayImages = DrawRegionsOnImagesWithText(None, cleanedImages, preparedImages, (255, 255, 255), 16,
                                            (0, 255, 0))
        #GenerateSizeDistributionAndAverageSize(cleanedImages)
        Output('Track', overlayImages)

    if overlay:
        overlayImages = DrawRegionsOnImages(cleanedImages, preparedImages, (255, 255, 255), 16,
                                            (0, 255, 0))
        Output('Overlay', overlayImages)

    if gif and outputPath is not None:
        MakeDirectory(outputPath)
        for name in outputImages:
            stacks = ConvertImagesToStacks(outputImages[name], pilImages) if batch else [
                outputImages[name]]
            for stack, original in zip(stacks, pilImages):
                path = Path(original.filename)
                SaveAsGIF(stack, outputPath / (path.stem + "_" + name.lower() + ".gif"))

    if colorLabeledOutput:
        Output('Color-Labeled', LabeledImagesToColoredImages(cleanedImages))

    if idLabeledOutput:
        overlayImages =DrawRegionsOnImagesWithText(None, cleanedImages, preparedImages, (255, 255, 255), 16,
                                            (0, 255, 0))
        Output('ID-Labeled', overlayImages)

    if run_class:
        class_model = tf.keras.layers.TFSMLayer(class_path, call_endpoint='serving_default') if (class_path is not None) else None
        overlayImages =DrawRegionsOnImagesWithText(class_model, cleanedImages, preparedImages, (255, 255, 255), 16,
                                            (0, 255, 0))
        Output('Classified', overlayImages)

    if outputPath is not None and computeProps:
        overlayImages = DrawRegionsAndSizeDistributions(cleanedImages, preparedImages, (255, 255, 255), 16,
                                            (0, 255, 0))
        Output('Size Distribution', overlayImages)


    return outputImages



def WindowPipeline(modelPath: Path, screenshots: List[np.ndarray]):
    
    classified_images = []
    model = YOLO(modelPath)
    for frame in screenshots:
        results = model(frame)
        classified_im = results[0].plot()
        classified_images.append(classified_im)

    return classified_images
    
         
