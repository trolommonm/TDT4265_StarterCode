import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path 
import matplotlib.pyplot as plt
import warnings
import click

import cv2
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image, scale_accross_batch_and_channels

from ssd import utils
import tops
from tops.config import instantiate
from tops.checkpointer import load_checkpoint

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# With reference to implementation of Class Activation Maps for Object Detection

# make sure python_grad_cam package is installed.
# To run, use python grad_cam.py configs/[config_name].py 

#creation of classes for identification
data_classes = ['background', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person','rider']

# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(data_classes), 3))

def predict(input_tensor, model, detection_threshold):
    outputs = model(input_tensor)

    pred_classes = [data_classes[i] for i in outputs[0][1].cpu().numpy()]
    pred_labels = outputs[0][1].cpu().numpy()
    pred_scores = outputs[0][2].detach().cpu().numpy()
    pred_bboxes = outputs[0][0].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            box = np.array([
                pred_bboxes[index][0] *1024,
                pred_bboxes[index][1] * 128,
                pred_bboxes[index][2] * 1024,
                pred_bboxes[index][3] * 128
            ]).astype(np.int32)
            boxes.append(box)
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices

def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_PLAIN, 0.5, color, 1, #edited the text
                    lineType=cv2.LINE_AA)
    return image

def renormalize_cam_in_bounding_boxes(boxes, labels, classes, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        images.append(img)
    
    renormalized_cam = np.max(np.float32(images), axis = 0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, eigencam_image_renormalized)
    return eigencam_image_renormalized 

@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.argument("img_dir", default="/work/datasets/tdt4265_2022/images/val/trip007_glos_Video00010_9.png", type=click.Path(exists=True, dir_okay=False, path_type=str))
def grad_cam(config_path: Path, img_dir: Path):

    cfg = utils.load_config(config_path)

    model = tops.to_cuda(instantiate(cfg.model))
    # print(model)
    model.eval()
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    model.load_state_dict(ckpt["model"])


    image = np.array(Image.open(img_dir))
    image_float_np = np.float32(image) / 255
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    image = transform(image)
    image = tops.to_cuda(image)
    transform = instantiate(cfg.data_val.gpu_transform)
    image = transform({"image": image})["image"]


    # Run the model and display the detections
    boxes, classes, labels, _ = predict(image, model, 0.4)

    target_layers = [model.feature_extractor.resnet_model.layer4]
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]

    # or a very fast alternative:
    cam = EigenCAM(model,
                target_layers, 
                use_cuda=torch.cuda.is_available(), 
                # reshape_transform=fasterrcnn_reshape_transform
                )
    cam.uses_gradients=False

    grayscale_cam = cam(image, targets=targets)

    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    renormalize_img = renormalize_cam_in_bounding_boxes(boxes, labels, classes, image_float_np, grayscale_cam)
    Image.fromarray(renormalize_img)
    plt.figure(figsize=(20,10))
    plt.imshow(renormalize_img)

    plt.savefig("outputs/sampleimg.png")
if __name__ == "__main__":
    grad_cam()