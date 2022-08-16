import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from omuti import GDB, TIFF, TIFF_CLIPPED


if __name__ == '__main__':
    with rasterio.open('scratch/subset_random.tif') as dataset:
        #rasterio.plot.show(dataset, with_bounds=True, cmap='gray')
        #arr = dataset.read(1, window=Window(10_000, 20_000, 5000, 4000))
        arr = dataset.read(1)
        print(arr.shape)
        print(arr.dtype)
        print(arr.min())
        print(arr.max())
        plt.imshow(arr, cmap='gray')
        plt.show()