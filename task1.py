import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display

from utils import get_dataset
from matplotlib.patches import Rectangle

# %matplotlib inline
dataset = get_dataset("./data/train/*.tfrecord", label_map="./experiments/label_map.pbtxt")


def display_instances(batch):
    plt.figure()
    img = batch['image'].numpy()
    classes = batch['groundtruth_classes'].numpy()
    boxes = batch['groundtruth_boxes'].numpy()

    print("------------------")
    print(len(boxes))
    # fit box to image
    img_h, img_w, channel = img.shape
    boxes[:, (0, 2)] *= img_h
    boxes[:, (1, 3)] *= img_w

    color = {1: [1, 0, 0], 2: [0, 1, 0], 4: [0, 0, 1]}

    # create a figure and axes objects(the objects that have plotting methods)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img.astype(np.uint8))

    # add colored boxes
    for box, category in zip(boxes, classes):
        anchor = (box[1], box[0])
        width = box[3] - box[1]
        height = box[2] - box[0]
        rec = Rectangle(anchor, width, height, edgecolor=color[category], facecolor='none')
        ax.add_patch(rec)

    # output
    plt.savefig("mygraph.png")
    return


dataset = dataset.shuffle(10)
dataset = dataset.take(1)
for batch in dataset:
    display_instances(batch)
    