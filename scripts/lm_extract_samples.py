import os
from collections import namedtuple

import fiona
from sklearn.model_selection import train_test_split
import geopandas
import rasterio.mask
from PIL import Image
from omuti import GDB, TIFF
from omuti.utils import convert_3D_2D
from tqdm import tqdm


OUTPUT_DIR = 'scratch/ml'
PADDING = 200  # padding to include beyond each labelled feature, on all sides
CSV_HEADER = 'image_path,xmin,ymin,xmax,ymax,label\n'

YoloBbox = namedtuple('YoloBbox', ['x_center', 'y_center', 'width', 'height'])   # All relative to image, [0, 1]
MyBbox = namedtuple('MyBbox', ['xmin', 'ymin', 'xmax', 'ymax'])  # pixel offsets from top left

if __name__ == '__main__':

    all_layers = fiona.listlayers(GDB)
    all_layers = all_layers[5:]
    for class_id, class_name in enumerate(all_layers):

        output_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

        f = open(f'{output_dir}/all.csv', 'w')
        f.write(CSV_HEADER)

        raster = rasterio.open('scratch/subset.tif')

        gdf = geopandas.read_file(GDB, layer=class_name)
        gdf.geometry = convert_3D_2D(gdf.geometry)
        series = gdf['geometry']  # GeoSeries

        with tqdm(total=len(series)) as pbar:
            pbar.set_description(f'Feature {class_name}')
            # For each shape found in the layer
            for i, shape in series.iteritems():
                out_image, out_transform = rasterio.mask.mask(raster, shape, pad=True, pad_width=PADDING, crop=True, filled=False)
                im = Image.fromarray(out_image.squeeze(0))
                image_basename = f'{i:04d}'
                image_path = os.path.abspath(f'{output_dir}/images/{image_basename}.png')
                im.save(image_path)

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
                bbox = MyBbox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

                f.write(f'{image_path},{bbox.xmin},{bbox.ymin},{bbox.xmax},{bbox.ymax},{class_name}\n')
                pbar.update(1)

        raster.close()
        f.close()

        all_lines = open(f'{output_dir}/all.csv', 'r').readlines()[1:]  # skip header!
        # some classes have no labelled data; skip these
        if not all_lines:
            continue

        training, testing = train_test_split(all_lines, train_size=0.8)
        training, validation = train_test_split(training, train_size=0.9)
        with open(f'{output_dir}/training.csv', 'w') as f:
            f.write(CSV_HEADER)
            f.writelines([f'{line}' for line in training])
        with open(f'{output_dir}/validation.csv', 'w') as f:
            f.write(CSV_HEADER)
            f.writelines([f'{line}' for line in validation])
        with open(f'{output_dir}/testing.csv', 'w') as f:
            f.write(CSV_HEADER)
            f.writelines([f'{line}' for line in testing])
