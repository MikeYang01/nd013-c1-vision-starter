import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import get_dataset

# %matplotlib inline
dataset = get_dataset(
    "./data/train/*.tfrecord", label_map="./experiments/label_map.pbtxt"
)


def display_instances(batch):
    plt.figure()
    img = batch["image"].numpy()
    boxes = batch["groundtruth_boxes"].numpy()

    # fit box to image
    img_h, img_w, channel = img.shape
    boxes[:, (0, 2)] = boxes[:, (0, 2)] * img_h
    boxes[:, (1, 3)] = boxes[:, (1, 3)] * img_w
    color = {1: "red", 2: "blue", 4: "green"}

    # create a figure and axes objects(the objects that have plotting methods)
    figure, axes = plt.subplots(1, figsize=(4, 4))

    # add colored boxes
    for box, classes in zip(boxes, batch["groundtruth_classes"].numpy()):
        anchor = (box[1], box[0])
        width = box[3] - box[1]
        height = box[2] - box[0]
        rec = patches.Rectangle(
            anchor,
            width,
            height,
            linewidth=1,
            edgecolor=color[classes],
            facecolor="none",
        )
        axes.add_patch(rec)

    # output
    axes.imshow(img)
    plt.savefig("mygraph.png")
    return


dataset = dataset.shuffle(10)
dataset = dataset.take(1)
for batch in dataset:
    display_instances(batch)
