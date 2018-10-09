# written by liwenxi
import os
import os.path
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import skimage.transform as transform
import skimage.io as io

count = 0;

def get_test_lists():

    test_file = os.path.join('VOCdevkit_test', 'VOC2007', 'ImageSets', 'Main')
    file_type = 'test.txt'
    # Set the groups in a dictionary.
    file_groups = []
    for root,dirs,files in os.walk(test_file):
        for i in files:
            if file_type in i:
                file_groups.append(i)



    return file_groups

def move_files(file_groups):

    # make the test folder
    os.mkdir(os.path.join('test'))

    for file in file_groups:
        print("Creating folder for %s" % (file))
        os.mkdir(os.path.join('test', file.replace('_test.txt', '')))
        test_file = os.path.join('VOCdevkit_test', 'VOC2007', 'ImageSets', 'Main', file)
        # print(test_file)
        op_file = np.loadtxt(test_file, dtype=str)
        for i in range(op_file.shape[0]):
            # print("test", op_file[i][0], op_file[i][1])
            if (str(op_file[i][1]) == '1'):
                print(str(op_file[i][1]))
                # print(op_file)
                scr = os.path.join('VOCdevkit_test', 'VOC2007', 'JPEGImages', op_file[i][0]+'.jpg')
                dest = os.path.join('test', file.replace('_test.txt', ''), op_file[i][0]+'.jpg')
                # print("Moving %s to %s" % (filename, dest))
                shutil.copy(scr, dest)

    print("Done.")

def read_xml():
    count = 0;
    test_file_path = os.path.join('VOCdevkit_test', 'VOC2007', 'Annotations')

    for root,dirs,files in os.walk(test_file_path):
        for i in files:
            get_region(i)


def get_region(test_file):
    test_file_path = os.path.join('VOCdevkit_test', 'VOC2007', 'Annotations')
    test_file_path = test_file_path + "/" + test_file
    doc = os.path.abspath(test_file_path)
    img_file = os.path.join('VOCdevkit_test', 'VOC2007', 'JPEGImages', test_file.replace(".xml", "")+'.jpg')
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
            if not os.path.exists(os.path.join("test", name)):
                # print("Creating folder for %s/%s" % (group, classname))
                os.makedirs(os.path.join("test", name))

            io.imsave("./test/"+name+"/"+str(len([lists for lists in os.listdir("./test/"+name) if os.path.isfile(os.path.join("./test/"+name, lists))])) +".jpg", nor_img)



def main():

    # Get the videos in groups so we can move them.
    group_lists = get_test_lists()

    # Move the files.
    move_files(group_lists)

def main2():
    read_xml()

if __name__ == '__main__':
    main2()
