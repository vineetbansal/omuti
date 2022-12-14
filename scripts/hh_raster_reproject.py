import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


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


if __name__ == '__main__':
    reproject_tif(
        input_tif='scratch/subset.tif',
        output_tif='scratch/subset_reprojected.tif',
        output_png='scratch/subset_reprojected.png'
    )
    reproject_tif(
        input_tif='scratch/subset_random.tif',
        output_tif='scratch/subset_random_reprojected.tif',
        output_png='scratch/subset_random_reprojected.png'
    )

