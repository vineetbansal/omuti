import os
from collections import namedtuple
from sklearn.model_selection import train_test_split
import geopandas
import rasterio.mask
from PIL import Image
from omuti import GDB, TIFF


OUTPUT_DIR = '../data/deepforest'
CLASSES = ['FarmBoundary1972', 'BigTree1972', 'Omuti1972', 'waterhole1972', 'FarmBoundary1943', 'BigTree1943',
           'waterhole1943', 'Cattlekraal1943', 'Cattlekraal1972', 'Omuti1943', 'OldOmuti', 'OldOmuti1943',
           'OldOmuti1972', 'Field1943']
CLASSES = ['Omuti1972']  # layer names in the gdb file that we wish to extract
PADDING = 200

YoloBbox = namedtuple('YoloBbox', ['x_center', 'y_center', 'width', 'height'])   # All relative to image, [0, 1]
DeepForestBbox = namedtuple('DeepForestBbos', ['xmin', 'ymin', 'xmax', 'ymax'])  # pixel offsets from top left

if __name__ == '__main__':

    header = 'image_path,xmin,ymin,xmax,ymax,label\n'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    f = open(f'{OUTPUT_DIR}/all.csv', 'w')
    f.write(header)

    all_images = []

    for class_id, class_name in enumerate(CLASSES):

        raster = rasterio.open('scratch/subset_reprojected.tif')

        os.makedirs(f'{OUTPUT_DIR}/{class_name}', exist_ok=True)

        gdf = geopandas.read_file(GDB, layer=class_name).to_crs(epsg=4326)
        series = gdf['geometry']  # GeoSeries

        # For each shape found in the layer
        for i, shape in series.iteritems():
            out_image, out_transform = rasterio.mask.mask(raster, shape, pad=True, pad_width=PADDING, crop=True, filled=False)
            im = Image.fromarray(out_image.squeeze(0))
            image_basename = f'{i:04d}'
            image_path = os.path.abspath(f'{OUTPUT_DIR}/{class_name}/{image_basename}.png')
            im.save(image_path)

            all_images.append(image_path)

            # Find the pixel width/height of the shape
            _minx, _miny, _maxx, _maxy = tuple(shape.bounds)
            _col_off1, _row_off1 = ~raster.transform * (_minx, _miny)
            _col_off2, _row_off2 = ~raster.transform * (_maxx, _maxy)
            _width, _height = _col_off2 - _col_off1, _row_off1 - _row_off2

            # By using rasterio.mask.mask with a padding, we know that the shape is centered w.r.t the image
            xmin = (im.width - _width) / 2.0
            xmax = (im.width + _width) / 2.0
            ymin = (im.height - _height) / 2.0
            ymax = (im.height + _height) / 2.0
            bbox = DeepForestBbox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

            f.write(f'{image_path},{bbox.xmin},{bbox.ymin},{bbox.xmax},{bbox.ymax},{class_name}\n')

        raster.close()
    f.close()

    all_lines = open(f'{OUTPUT_DIR}/all.csv', 'r').readlines()[1:]  # skip header!
    training, validation = train_test_split(all_lines)
    with open(f'{OUTPUT_DIR}/training.csv', 'w') as f:
        f.write(header)
        f.writelines([f'{line}' for line in training])
    with open(f'{OUTPUT_DIR}/validation.csv', 'w') as f:
        f.write(header)
        f.writelines([f'{line}' for line in validation])
