import matplotlib.pyplot as plt
from utils import get_dataset
import numpy as np
import matplotlib
# %matplotlib inline
dataset = get_dataset("./data/train/*.tfrecord", label_map="./experiments/label_map.pbtxt")


def get_box_num(batch):
    # count = len(boxes) 
    # print(count)
    color = {1: "red", 2: "blue", 4: "green"}
    category_list = []

    box_v = 0
    box_p = 0
    box_c = 0

    # add colored boxes
    for box, category in zip(batch["groundtruth_boxes"].numpy(), batch["groundtruth_classes"].numpy()):
        if category == 1:
            box_v = box_v +1
        if category == 2:
            box_p = box_p +1
        if category == 4:
            box_c = box_c +1
    
    category_list.append(box_v)
    category_list.append(box_p)
    category_list.append(box_c)
    return category_list


dataset = dataset.shuffle(100)
dataset = dataset.take(100)


box_v_all = 0
box_p_all = 0
box_c_all = 0
    
for batch in dataset:
    box_v_all = box_v_all + get_box_num(batch)[0]
    box_p_all = box_p_all + get_box_num(batch)[1]
    box_c_all = box_c_all + get_box_num(batch)[2]

data = [];
data.append(box_v_all)
data.append(box_p_all)
data.append(box_c_all)

label_list = ['vehicle', 'pedestrian', 'cyclist']
p1 = plt.bar(label_list, data)
plt.bar_label(p1, label_type='edge')
plt.show()
plt.savefig("mygraph.png")