import os
from collections import namedtuple
import xml.etree.ElementTree as ET


VOC_ROOT = '/data/testbed/yoltv4/darknet/data/voc'
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


PascalBbox = namedtuple('PascalBbox', ['xmin', 'ymin', 'xmax', 'ymax'])
YoloBbox = namedtuple('YoloBbox', ['x_center', 'y_center', 'width', 'height'])


def pascal_to_yolo_bbox(im_width, im_height, pascal_bbox):

    x = (pascal_bbox.xmin + pascal_bbox.xmax) / 2.0
    y = (pascal_bbox.ymin + pascal_bbox.ymax) / 2.0
    w = pascal_bbox.xmax - pascal_bbox.xmin
    h = pascal_bbox.ymax - pascal_bbox.ymin

    x = x / im_width
    w = w / im_width
    y = y / im_height
    h = h / im_height

    return YoloBbox(x_center=x, y_center=y, width=w, height=h)


def convert_annotation(year, image_id, classes):

    with open(f'{VOC_ROOT}/VOCdevkit/VOC{year}/Annotations/{image_id}.xml') as f:
        tree = ET.parse(f)

    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(f'{VOC_ROOT}/VOCdevkit/VOC{year}/labels/{image_id}.txt', 'w') as f:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')

            bbox = PascalBbox(
                xmin=float(xmlbox.find('xmin').text),
                xmax=float(xmlbox.find('xmax').text),
                ymin=float(xmlbox.find('ymin').text),
                ymax=float(xmlbox.find('ymax').text)
            )
            bbox = pascal_to_yolo_bbox(w, h, bbox)
            f.write(f'{cls_id} {bbox.x_center} {bbox.y_center} {bbox.width} {bbox.height}\n')


if __name__ == '__main__':

    sets = (('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'))

    for year, phase in sets:
        os.makedirs(f'{VOC_ROOT}/VOCdevkit/VOC{year}/labels/', exist_ok=True)
        image_ids = open(f'{VOC_ROOT}/VOCdevkit/VOC{year}/ImageSets/Main/{phase}.txt').read().strip().split()

        with open(f'{VOC_ROOT}/{year}_{phase}.txt', 'w') as f:
            for image_id in image_ids:
                f.write(f'{VOC_ROOT}/VOCdevkit/VOC{year}/JPEGImages/{image_id}.jpg\n')
                convert_annotation(year, image_id, CLASSES)