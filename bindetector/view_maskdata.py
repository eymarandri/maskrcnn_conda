
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys

import numpy as np
from skimage import io
import skimage.io
import argparse
import scipy
import random
import colorsys

from skimage.measure import find_contours
import matplotlib.pyplot as plt


ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.model import load_image_gt

parser = argparse.ArgumentParser(description='set inputdirs')
parser.add_argument('-id', '--input_dir', type=str, help='Base directory containing original images and masks', required=True)
args = parser.parse_args()

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    print(colors[0])
    return colors

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()

def fig_to_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    return buf

def imglocator(dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    dataset_dir = os.path.join(dataset_dir, subset)
    image_ids = next(os.walk(dataset_dir))[1]
    print(image_ids)
    #masks=[]
    images=[]
    for image_id in image_ids:
        curr_path=os.path.join(dataset_dir, image_id)

        rgb_path=os.path.join(curr_path, 'images')
        mask_path = os.path.join(curr_path, 'masks')
        if not os.path.exists(rgb_path) or not os.path.isdir(mask_path):
            print("WTF")
            continue
        else:
            print('\n')
            mask = []
            maskp = next(os.walk(mask_path))[2][0]
            maskp=os.path.join(mask_path, maskp)
            imagep=next(os.walk(rgb_path))[2][0]
            imagep=os.path.join(rgb_path, imagep)
            print(imagep)
            mask=skimage.io.imread(maskp)
            #masks.append(mask)
            rgbimg = skimage.io.imread(imagep)
            colors=random_colors(10,bright=True)

            #for f in next(os.walk(mask_path))[2]:
                #if f.endswith(".png"):
            #        m = skimage.io.imread(os.path.join(mask_path, f)).astype(np.bool)
            #        mask.append(m)
            #mask = np.stack(mask, axis=-1)
            #print(mask.size)

            image=apply_mask(rgbimg,mask,colors[0])
            images.append(image)
    display_images(images[0:20], cols=6)


        # Save image with masks
        #visualize.display_instances(
        #    image, r['rois'], r['masks'], r['class_ids'],
        #    dataset.class_names, r['scores'],
        #    show_bbox=True, show_mask=True,
        #    title="Predictions")
        #plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))


check=imglocator(args.input_dir,'test')
dataset_dir = os.path.join(args.input_dir, 'test')
for images_or_masks in ['images','masks']:
    dir = os.path.join(dataset_dir, images_or_masks)
    print(dir)
    if not os.path.exists(dir) or not os.path.isdir(dir):
        continue

    for filename in os.listdir(dir):
        print(filename)
        #filepath = os.path.join(dir, filename)
        #img = io.imread(filepath)


