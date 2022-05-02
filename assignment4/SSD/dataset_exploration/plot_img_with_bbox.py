import sys, os
import torch
import matplotlib.pyplot as plt
from vizer.draw import draw_boxes

def visualize_img(image, boxes, labels, cfg, figsize=(20,20), to_print=True, to_save=False, save_dir=None):
    image_mean = torch.tensor(cfg.data_train.gpu_transform.transforms[0].mean).view(1, 3, 1, 1)
    image_std = torch.tensor(cfg.data_train.gpu_transform.transforms[0].std).view(1, 3, 1, 1)
    
    im = image.clone()
    im = (im * image_std + image_mean)
    im = (im * 255).byte()
    im = im[0]
    box = boxes[0].clone()
    box[:, [0,2]] *= im.shape[-1]
    box[:, [1,3]] *= im.shape[-2]
    
    im = im.permute(1, 2, 0).cpu().numpy()
    im = draw_boxes(im, box.cpu().numpy(), labels[0].cpu().numpy().tolist(), class_name_map=cfg.label_map)
    
    if to_print:
        plt.figure(figsize=figsize)
        plt.imshow(im)
        
    if to_save:
#         if save_dir is None:
#             sys.path.append(os.path.dirname(os.getcwd()))
#             save_dir = os.path.join(os.getcwd(), "dataset_exploration/visualizations/")
#             print(f"Saving to default dir: {save_dir}")
            

        
        os.makedirs(save_dir, exist_ok=True)
        files = [y for x in os.walk(save_dir) for y in x[2]]
        curr_idx = len(files)

        file_name = f"visualization_{curr_idx:04d}.png"
        save_dir = os.path.join(save_dir, file_name)
        plt.imsave(save_dir, im)
    
    return im

def visualize_img_no_box(image, cfg, figsize=(20,20), to_print=True):
    image_mean = torch.tensor(cfg.data_train.gpu_transform.transforms[0].mean).view(1, 3, 1, 1)
    image_std = torch.tensor(cfg.data_train.gpu_transform.transforms[0].std).view(1, 3, 1, 1)
    
    im = image.clone()
    im = (im * image_std + image_mean)
    im = (im * 255).byte()
    im = im[0]
    im = im.permute(1, 2, 0).cpu().numpy()
    
    if to_print:
        plt.figure(figsize=figsize)
        plt.imshow(im)
        
    return im