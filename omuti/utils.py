import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon, MultiPolygon


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


def reproject_tif(input_tif, output_tif, dst_crs='epsg:4326', output_png=None):
    with rasterio.open(input_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                destination, dst_transform = reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

    if output_png is not None:
        with rasterio.open(output_tif) as dataset_in:
            arr = dataset_in.read()
            profile = dataset_in.profile
            profile['driver'] = 'PNG'

            with rasterio.open(output_png, 'w', **profile) as dataset_out:
                dataset_out.write(arr)


def convert_3D_2D(geometry):
    '''
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
    '''
    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == 'Polygon':
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == 'MultiPolygon':
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(MultiPolygon(new_multi_p))
    return new_geo
