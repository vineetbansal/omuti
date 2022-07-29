from osgeo import gdal
import matplotlib.pyplot as plt
from omuti import GDB, TIFF


if __name__ == '__main__':
    dataset = gdal.Open(TIFF, gdal.GA_ReadOnly)
    print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                                dataset.GetDriver().LongName))
    print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                        dataset.RasterYSize,
                                        dataset.RasterCount))
    print("Projection is {}".format(dataset.GetProjection()))
    geotransform = dataset.GetGeoTransform()
    if geotransform:
        print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

    band = dataset.GetRasterBand(1)  # 1-indexed, 1..dataset.RasterCount
    print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

    min = band.GetMinimum()
    max = band.GetMaximum()
    if not min or not max:
        (min, max) = band.ComputeRasterMinMax(True)
    print("Min={:.3f}, Max={:.3f}".format(min, max))

    if band.GetOverviewCount() > 0:
        print("Band has {} overviews".format(band.GetOverviewCount()))

    if band.GetRasterColorTable():
        print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

    # The original TIFF is huge - only extract a small window for demonstration
    im = band.ReadAsArray(win_xsize=500, win_ysize=400)
    print(im.shape)

    x = im
    print(x.shape)

    plt.imshow(x)
    plt.show()