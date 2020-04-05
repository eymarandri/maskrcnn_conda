"""
Mask R-CNN

------------------------------------------------------------

    # Train a new model starting from ImageNet weights
    python3 bindetector.py train --dataset=/path/to/dataset --subset=train --weights=coco

    # Train a new model starting from specific weights file
    python3 bindetector.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 bindetector.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 bindetector.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>

    Added:
    python3 bindetector.py singledetect --image=.... --weights=last

    python3 bindetector.py splash, crap
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
CLASS_NAME="Bin" #or graybox
# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/bindetector/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
"""
VAL_IMAGE_IDS = [
    "rgb_0_inst_0_class_objects_seg_0_augm_0",
    "rgb_0_inst_0_class_objects_seg_0_orig",
    "rgb_0_inst_0_class_objects_seg_1_augm_0",
    "rgb_0_inst_0_class_objects_seg_1_orig",
    "rgb_0_inst_1_class_objects_seg_0_augm_0",
    "rgb_0_inst_1_class_objects_seg_0_orig",
    "rgb_19_inst_0_class_objects_seg_0_augm_0",
    "rgb_19_inst_0_class_objects_seg_0_orig",
    "rgb_19_inst_0_class_objects_seg_1_augm_0",
    "rgb_19_inst_0_class_objects_seg_1_orig",
    "rgb_20_inst_0_class_objects_seg_0_augm_0",
    "rgb_20_inst_0_class_objects_seg_0_orig",
    "rgb_20_inst_0_class_objects_seg_1_augm_0",
    "rgb_20_inst_0_class_objects_seg_1_orig",
    "rgb_21_inst_0_class_objects_seg_0_augm_0",
    "rgb_21_inst_0_class_objects_seg_0_orig",
    "rgb_21_inst_0_class_objects_seg_1_augm_0",
    "rgb_21_inst_0_class_objects_seg_1_orig",
    "rgb_22_inst_0_class_objects_seg_0_augm_0",
    "rgb_22_inst_0_class_objects_seg_0_orig",
    "rgb_22_inst_0_class_objects_seg_1_augm_0",
    "rgb_22_inst_0_class_objects_seg_1_orig",
    "rgb_23_inst_0_class_objects_seg_0_augm_0",
    "rgb_23_inst_0_class_objects_seg_0_orig",
    "rgb_23_inst_0_class_objects_seg_1_augm_0",
    "4ac1a641-1bea-4d57-a96e-99d018270a84_inst_13_class_graybox_seg_0_orig",
    "8e653167-e9a8-43d1-88ca-82a3505e2777_inst_3_class_graybox_seg_1_orig",
    "8e653167-e9a8-43d1-88ca-82a3505e2777_inst_4_class_graybox_seg_1_orig",
    "11ad301e-af27-4f63-9f70-98e7cba8b017_inst_2_class_graybox_seg_0_orig",
    "86fa45d5-0a9f-4fe9-86b5-2d474188689a_inst_0_class_graybox_seg_0_augm_0",
    "2342ffdc-711d-4851-92ac-447cebe9c30c_inst_2_class_graybox_seg_1_augm_0",
    "rgb_6tupe_inst_0_class_graybox_seg_0_orig",
    "rgb_13tupe_inst_0_class_graybox_seg_1_orig",
    #"5fda43e8-6370-4fff-a465-65b8c7fb4432_inst_16_class_graybox_seg_1_orig",

]
"""
VAL_IMAGE_IDS = [
    "5c9404cb-6e81-4e24-8553-24deecc3c793_inst_0_class_Bin_seg_0_orig",
    "59045505-ad7d-4537-98f2-552b5682dcec_inst_1_class_Bin_seg_1_orig",
    "5734f100-d6df-4780-b0ab-d82a94b765e7_inst_0_class_Bin_seg_0_augm_0",
    "4c9ebbc3-f484-4b56-98c7-fde00df79f3f_inst_0_class_Bin_seg_0_orig",
    "418a56db-2c95-4af3-a2bd-5ba6885ba9ee_inst_0_class_Bin_seg_0_augm_0",
    "2e518840-c5d9-4cd8-9501-174a9182bab0_inst_0_class_Bin_seg_1_augm_1",
    "16bbb97e-5908-43f8-9af0-0f2c90b4a04a_inst_0_class_Bin_seg_0_augm_0",
    "09d85c8e-706e-4078-916b-c894babb3228_inst_1_class_Bin_seg_0_augm_3",
    "09d40f43-82d0-4543-a6bb-be880dd16cdb_inst_0_class_Bin_seg_0_orig",
    "09c67a2a-a976-4f5f-904c-df45a8c0619b_inst_1_class_Bin_seg_1_orig",
    "08b96752-e44a-46a1-8a1c-ea0212285b56_inst_1_class_Bin_seg_1_orig",
    "07423e49-bc26-466e-b24d-e0df2b8c20f8_inst_0_class_Bin_seg_0_orig",
    "071dc781-d691-47fe-99ce-a2f2b46984ac_inst_1_class_Bin_seg_0_augm_4",
    "067269ea-5f34-4d53-904d-42f714b0391a_inst_2_class_Bin_seg_1_augm_2",
    "06318e29-06b0-4592-8953-f689da510f29_inst_0_class_Bin_seg_0_orig",
    "055d7e1f-1dd1-4a21-a2ae-0e0bb60632d0_inst_0_class_Bin_seg_1_orig",
    "rgb_301_inst_0_class_graybox_seg_0_orig",
    "rgb_301_inst_0_class_graybox_seg_0_augm_0",
    "8e653167-e9a8-43d1-88ca-82a3505e2777_inst_3_class_graybox_seg_1_augm_0",
    "rgb_2tupe_inst_0_class_graybox_seg_1_augm_0",
    "rgb_10tupe_inst_0_class_graybox_seg_1_augm_0",
    "fceaac94-fecf-482c-b127-c65cbf5d786c_inst_0_class_graybox_seg_0_orig",
    "8e653167-e9a8-43d1-88ca-82a3505e2777_inst_0_class_graybox_seg_1_orig",
    "86fa45d5-0a9f-4fe9-86b5-2d474188689a_inst_0_class_graybox_seg_0_augm_0",
    "5b771c95-bc1b-401c-94e7-ed690cb8f63a_inst_1_class_graybox_seg_0_augm_0",
    "5b771c95-bc1b-401c-94e7-ed690cb8f63a_inst_0_class_graybox_seg_1_augm_0",
    "46884373-51f9-452c-b841-8b78ceb6ea69_inst_0_class_graybox_seg_1_augm_0",
    "2342ffdc-711d-4851-92ac-447cebe9c30c_inst_0_class_graybox_seg_1_orig",
    "11ad301e-af27-4f63-9f70-98e7cba8b017_inst_0_class_graybox_seg_0_augm_0",
    "0bdb2ac4-156e-4b39-8fa3-9cd13893cfb3_inst_0_class_graybox_seg_0_orig",
    #"5fda43e8-6370-4fff-a465-65b8c7fb4432_inst_16_class_graybox_seg_1_orig",

]

############################################################
#  Configurations
############################################################

class BinSegmentationConfig(Config):
    """Configuration for training on the BinSegmentation segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = CLASS_NAME#"graybox"
    GPU_COUNT = 1
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + BinSegmentation

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 374
    #STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between BinSegmentation and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512 #512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    #MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class BinSegmentationInferenceConfig(BinSegmentationConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class BinSegmentationDataset(utils.Dataset):

    def load_BinSegmentation(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset BinSegmentation, and the class BinSegmentation
        self.add_class(CLASS_NAME, 1, CLASS_NAME)

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        # class_files = list(map(
        #     (lambda x: os.path.basename(x)),
        #     glob.glob('{}/*_class_{}*'.format(args.input_dir, "graybox"))))
        #if subset=="trainval":
        #    assert subset in ["train", "val"]
        #    dataset_dir = os.path.join(dataset_dir, subset)
        #else:

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        image_ids = next(os.walk(dataset_dir))[1]
        if subset=="val":
            print("VALIDATUIN size !!!!!!!!!!!!!!!!!!!!!!!: ")
            print(len(image_ids))
        if subset == "train":
            print("train  LENGTH!!!!!!!!!!!!!!!!!: ")
            print(len(image_ids))

        """
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = next(os.walk(dataset_dir))[1]
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))
        """

        # Add images
        for image_id in image_ids:
            self.add_image(
                CLASS_NAME,
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]

        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == CLASS_NAME:
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = BinSegmentationDataset()
    #dataset_train.load_BinSegmentation(dataset_dir, subset)
    dataset_train.load_BinSegmentation(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BinSegmentationDataset()
    dataset_val.load_BinSegmentation(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    #augmentation = iaa.SomeOf((0, 2), [
    #    iaa.Fliplr(0.5),
    #    iaa.Flipud(0.5),
    #    iaa.OneOf([iaa.Affine(rotate=90),
    #               iaa.Affine(rotate=180),
    #               iaa.Affine(rotate=270)]),
    #    iaa.Multiply((0.8, 1.5)),
    #    iaa.GaussianBlur(sigma=(0.0, 5.0))
    #])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #if len(physical_devices) > 0:
    #    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #    tf.config.experimental.set_memory_growth(physical_devices[1], True)
    #config.GPU_COUNT=2

    #model.train(dataset_train, dataset_val,
    #            learning_rate=config.LEARNING_RATE,
    #            epochs=30,#20
    #            #augmentation=augmentation,
    #            layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,#40
                #augmentation=augmentation,
                layers='4+')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detectsingleimg(model, image_path=None):
    """Run detection on images in the given directory."""
    #print("Running on {}".format(dataset_dir))
    assert image_path
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        print('Found ', len(r['masks']))
        print('Found ', len(r['rois']))
        for rio in r['rois']:
            print(rio)
        # Color splash
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            CLASS_NAME, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions")
        file_name = "splashPredict_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
        plt.savefig("{}.png".format(file_name,image))
        #skimage.io.imsave(file_name, splash)
    print("Saved to ", file_name)




def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = BinSegmentationDataset()
    dataset.load_BinSegmentation(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)

def detectsplash(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = BinSegmentationDataset()
    dataset.load_BinSegmentation(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        # Color splash
        #splash = color_splash(image, r['masks'])
        # Save output
        #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        #skimage.io.imsave(file_name, splash)
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None):
    assert image_path
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        print('Found ', len(r['masks']))
        print('Found ', len(r['rois']))
        for rio in r['rois']:
            print(rio)
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    print("Saved to ", file_name)




############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for box segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'singledetect' or 'detect' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--classname', required=False,
                        default=CLASS_NAME,
                        metavar="Class names",
                        help='Class names, str')
    args = parser.parse_args()
    print("CLASS NAME!:",args.classname)
    CLASS_NAME=args.classname

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"
    elif args.command == "splash":
        assert args.image
    elif args.command == "singledetect":
        assert args.image\


    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BinSegmentationConfig()
    else:
        config = BinSegmentationInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)


    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image)
    elif args.command == "singledetect":
        detectsingleimg(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
