# written by liwenxi
import os
import os.path
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import skimage.transform as transform
import skimage.io as io

count = 0;

def get_train_lists():

    train_file = os.path.join('VOCdevkit', 'VOC2007', 'ImageSets', 'Main')
    file_type = 'trainval.txt'
    # Set the groups in a dictionary.
    file_groups = []
    for root,dirs,files in os.walk(train_file):
        for i in files:
            if file_type in i:
                file_groups.append(i)



    return file_groups

def move_files(file_groups):

    # make the train folder
    os.mkdir(os.path.join('train'))

    for file in file_groups:
        print("Creating folder for %s" % (file))
        os.mkdir(os.path.join('train', file.replace('_trainval.txt', '')))
        train_file = os.path.join('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', file)
        # print(train_file)
        op_file = np.loadtxt(train_file, dtype=str)
        for i in range(op_file.shape[0]):
            # print("test", op_file[i][0], op_file[i][1])
            if (str(op_file[i][1]) == '1'):
                print(str(op_file[i][1]))
                # print(op_file)
                scr = os.path.join('VOCdevkit', 'VOC2007', 'JPEGImages', op_file[i][0]+'.jpg')
                dest = os.path.join('train', file.replace('_trainval.txt', ''), op_file[i][0]+'.jpg')
                # print("Moving %s to %s" % (filename, dest))
                shutil.copy(scr, dest)

    print("Done.")

def read_xml():
    count = 0;
    train_file_path = os.path.join('VOCdevkit', 'VOC2007', 'Annotations')

    for root,dirs,files in os.walk(train_file_path):
        for i in files:
            get_region(i)


def get_region(train_file):
    train_file_path = os.path.join('VOCdevkit', 'VOC2007', 'Annotations')
    train_file_path = train_file_path + "/" + train_file
    doc = os.path.abspath(train_file_path)
    img_file = os.path.join('VOCdevkit', 'VOC2007', 'JPEGImages', train_file.replace(".xml", "")+'.jpg')
    img = io.imread(img_file)
    # print(doc)

    tree = ET.parse(doc)
    root = tree.getroot()
    region = []

    for child in root:
        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0
        # print("遍历root的下一层", child.tag, "----", child.attrib)
        if child.tag == "object":
            for content in child:
                if content.tag == "name":
                    name = content.text
                if content.tag == "bndbox":

                    for i in content:
                        if i.tag == "xmin":
                            xmin = int(i.text)
                        if i.tag == "xmax":
                            xmax = int(i.text)
                        if i.tag == "ymin":
                            ymin = int(i.text)
                        if i.tag == "ymax":
                            ymax = int(i.text)
            print(name, xmin, xmax, ymin, ymax)
            # print(type(xmin))
            nor_img = transform.resize(img[ymin:ymax, xmin:xmax, :], (224, 224))
            # Check if this class exists.
            if not os.path.exists(os.path.join("train", name)):
                # print("Creating folder for %s/%s" % (group, classname))
                os.makedirs(os.path.join("train", name))

            io.imsave("./train/"+name+"/"+str(len([lists for lists in os.listdir("./train/"+name) if os.path.isfile(os.path.join("./train/"+name, lists))])) +".jpg", nor_img)



def main():

    # Get the videos in groups so we can move them.
    group_lists = get_train_lists()

    # Move the files.
    move_files(group_lists)

def main2():
    read_xml()

if __name__ == '__main__':
    main2()
