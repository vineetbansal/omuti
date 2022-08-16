PASCAL Annotation format (one per image) is of the form:
---------------
<annotation>
        <folder>VOC2012</folder>
        <filename>2008_000008.jpg</filename>
        <source>
                <database>The VOC2008 Database</database>
                <annotation>PASCAL VOC2008</annotation>
                <image>flickr</image>
        </source>
        <size>
                <width>500</width>
                <height>442</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
                <name>horse</name>
                <pose>Left</pose>
                <truncated>0</truncated>
                <occluded>1</occluded>
                <bndbox>
                        <xmin>53</xmin>
                        <ymin>87</ymin>
                        <xmax>471</xmax>
                        <ymax>420</ymax>
                </bndbox>
                <difficult>0</difficult>
        </object>
        <object>
                <name>person</name>
                <pose>Unspecified</pose>
                <truncated>1</truncated>
                <occluded>0</occluded>
                <bndbox>
                        <xmin>158</xmin>
                        <ymin>44</ymin>
                        <xmax>289</xmax>
                        <ymax>167</ymax>
                </bndbox>
                <difficult>0</difficult>
        </object>
</annotation>
------------

origin is top left
--->  x increases from 0 to width
|
|
v

y increases from 0 to height

Yolo Bounding box format
-----------------------
x_center, y_center, width, height
where all of the above are relative to the width/height of the image, so all between 0 and 1


./darknet detect cfg/yolov4.cfg weights/yolov4.weights data/dog.jpg
./darknet detector test cfg/coco.data cfg/yolov4.cfg weights/yolov4.weights data/dog.jpg

/data/testbed/yoltv4/darknet/data/voc/VOCdevkit/VOC2007/labels/003501.txt
6 0.505 0.40326975476839233 0.582 0.7629427792915531
14 0.596 0.5694822888283378 0.156 0.7247956403269754
14 0.185 0.5667574931880108 0.226 0.6757493188010899

<object-class> <x> <y> <width> <height>

6 is bus (https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)

Training
---------

./darknet detector train /data/omuti/darknet/omuti.data /data/omuti/darknet/omuti.cfg weights/yolov4.conv.137

Detection
---------

./darknet detector test /data/omuti/darknet/omuti.data /data/omuti/darknet/omuti.cfg /data/omuti/darknet/omuti_final.weights /data/omuti/data/yolo/Omuti1972/0000.png --thresh 0.1

./darknet detector test /data/omuti/darknet/omuti.data /data/omuti/darknet/omuti.cfg /data/omuti/darknet/omuti_final.weights /data/omuti/scripts/scratch/subset_random_reprojected.png --thresh 0.1