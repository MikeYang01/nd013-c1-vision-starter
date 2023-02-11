import matplotlib.pyplot as plt
from utils import get_dataset

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


count_list_v= []
count_list_p= []
count_list_c= []

for batch in dataset:
    count_list_v.append(get_box_num(batch)[0])
    count_list_p.append(get_box_num(batch)[1])
    count_list_c.append(get_box_num(batch)[2])

numBins = [0,10,20,30,40,50,60,70,80,90,100]

plt.hist(count_list_v, numBins, color='red')
plt.hist(count_list_p, numBins, color='blue')
plt.hist(count_list_c, numBins, color='green')


plt.show()
plt.savefig("mygraph.png")