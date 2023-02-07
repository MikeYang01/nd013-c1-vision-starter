import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display

from utils import get_dataset
from matplotlib.patches import Rectangle

# %matplotlib inline
dataset = get_dataset("./data/train/*.tfrecord", label_map="./experiments/label_map.pbtxt")


def get_box_num(batch):
    classes = batch['groundtruth_classes'].numpy()
    boxes = batch['groundtruth_boxes'].numpy()

    count = len(boxes) 
    print(count)
    return count


dataset = dataset.shuffle(100)
dataset = dataset.take(100)

count_list = []
for batch in dataset:
    count_list.append(get_box_num(batch))

numBins = [0,10,20,30,40,50,60,70,80,90,100]

plt.hist(count_list, numBins)
plt.title('Groundtruth Boxes Distribution')

plt.show()
plt.savefig("mygraph.png")
    
    
