import os
from collections import namedtuple
from sklearn.model_selection import train_test_split
import geopandas
import rasterio.mask

from PIL import Image


GDB = '/data/projects/kreike/data/KreikeSampleExtractedDataNam52022.gdb/'
OUTPUT_DIR = 'out'
# CLASSES = ['Omuti1972']  # layer names in the gdb file that we wish to extract
CLASSES = ['FarmBoundary1972', 'BigTree1972', 'Omuti1972', 'waterhole1972', 'FarmBoundary1943', 'BigTree1943',
           'waterhole1943', 'Cattlekraal1943', 'Cattlekraal1972', 'Omuti1943', 'OldOmuti', 'OldOmuti1943',
           'OldOmuti1972', 'Field1943']

YoloBbox = namedtuple('YoloBbox', ['x_center', 'y_center', 'width', 'height'])


if __name__ == '__main__':

    all_images = []

    with open(f'{OUTPUT_DIR}/names.txt', 'w') as f:
        f.writelines([f'{k}\n' for k in CLASSES])

    for class_id, class_name in enumerate(CLASSES):

        raster = rasterio.open('subset_reprojected.tif')

        os.makedirs(f'{OUTPUT_DIR}/{class_name}/images', exist_ok=True)

        gdf = geopandas.read_file(GDB, layer=class_name).to_crs(epsg=4326)
        series = gdf['geometry']  # GeoSeries

        # For each shape found in the layer
        for i, shape in series.iteritems():
            out_image, out_transform = rasterio.mask.mask(raster, shape, pad=True, pad_width=50, crop=True, filled=False)
            im = Image.fromarray(out_image.squeeze(0))
            image_basename = f'{i:04d}'
            image_path = os.path.abspath(f'{OUTPUT_DIR}/{class_name}/images/{image_basename}.png')
            im.save(image_path)

            all_images.append(image_path)

            # Find the pixel width/height of the shape
            _minx, _miny, _maxx, _maxy = tuple(shape.bounds)
            _col_off1, _row_off1 = ~raster.transform * (_minx, _miny)
            _col_off2, _row_off2 = ~raster.transform * (_maxx, _maxy)
            _width, _height = _col_off2 - _col_off1, _row_off1 - _row_off2

            # By using rasterio.mask.mask with a padding, we know that the shape is centered w.r.t the image
            # Yolo expects all values in the bounding box to be specified as the fraction of the total image
            # in the following order, so that all values lie between 0 and 1
            #    class_id, x_center, y_center, width, height
            bbox = YoloBbox(x_center=0.5, y_center=0.5, width=_width/im.width, height=_height/im.height)

            with open(f'{OUTPUT_DIR}/{class_name}/images/{image_basename}.txt', 'w') as f:
                f.write(f'{class_id} {bbox.x_center} {bbox.y_center} {bbox.width} {bbox.height}\n')

        raster.close()

        training, validation = train_test_split(all_images)
        with open(f'{OUTPUT_DIR}/training.txt', 'w') as f:
            f.writelines([f'{img}\n' for img in training])
        with open(f'{OUTPUT_DIR}/validation.txt', 'w') as f:
            f.writelines([f'{img}\n' for img in validation])