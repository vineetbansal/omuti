import os.path
import warnings
import fiona
import geopandas
import numpy as np
import rasterio
import rasterio.plot
from rasterio.windows import Window
import matplotlib.pyplot as plt
from omuti import GDB, TIFF


def subset(input_tif, output_tif, gdf, band_index=1, random=False, random_h=1000, random_w=5000):
    with rasterio.open(input_tif) as raster:
        if raster.crs.wkt != gdf.crs:
            raise AssertionError

        _minx, _miny, _maxx, _maxy = tuple(gdf.total_bounds)

        if random:
            _minx = np.random.randint(_minx, _maxx - random_w)
            _maxx = _minx + random_w
            _miny = np.random.randint(_miny, _maxy - random_h)
            _maxy = _miny + random_h

        _col_off1, _row_off1 = ~raster.transform * (_minx, _miny)
        _col_off2, _row_off2 = ~raster.transform * (_maxx, _maxy)

        _width, _height = _col_off2-_col_off1, _row_off1-_row_off2

        window = Window(
            float(_col_off1),
            float(_row_off2),
            float(_width),
            float(_height)
        )

        band = raster.read(
            band_index,
            window=window
        )

    with rasterio.open(
            output_tif,
            mode='w',
            driver='GTiff',
            height=band.shape[0],
            width=band.shape[1],
            count=1,
            dtype=band.dtype,
            crs=raster.crs.wkt,
            transform=rasterio.windows.transform(window, raster.transform),
    ) as new_dataset:
        new_dataset.write(band, indexes=1)


if __name__ == '__main__':

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    # Set index to the country name (unique)
    world = world.set_index('name', drop=False)

    africa = world[(world['continent'] == 'Africa')]
    # Some attributes, like area/centroid/boundary, are available to us for 'world', since it has a geometry column
    print(africa.area['Angola'])  # world.area is a Series, indexed by the same index as the DataFrame (country)

    # The active geometry of a GeoDataFrame gives us a GeoSeries that we can plot/explore
    # africa.plot()  # explore() works in jupyter
    roi = africa[((africa['name'] == 'Angola') | (africa['name'] == 'Namibia'))]

    layers = fiona.listlayers(GDB)
    print(layers)
    gdf = geopandas.read_file(GDB, layer='Omuti1972')

    roi = roi.to_crs(gdf.crs)
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey='all', sharex='all')
    ax.ticklabel_format(useOffset=False, style='plain')

    subset_tiff = 'scratch/subset.tif'
    subset(TIFF, subset_tiff, gdf)
    with rasterio.open(subset_tiff, 'r') as raster:
        rasterio.plot.show(raster, with_bounds=True, ax=ax)
        #roi.plot(ax=ax, color='lightgrey', edgecolor=None)
        gdf.plot(ax=ax, color='blue')

    subset_tiff = 'scratch/subset_random.tif'
    subset(TIFF, subset_tiff, gdf, random=True)
    with rasterio.open(subset_tiff, 'r') as raster:
        rasterio.plot.show(raster, with_bounds=True, ax=ax)
        #roi.plot(ax=ax, color='lightgrey', edgecolor=None)
        gdf.plot(ax=ax, color='blue')

    plt.show()
