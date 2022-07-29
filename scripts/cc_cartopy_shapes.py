import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt

# Downloaded from gadm.org
fname = '../data/gadm41_FRA/gadm41_FRA_0.shp'

x = shpreader.Reader(fname)  # FionaReader
adm1_shapes = list(x.geometries())  # list of MultiPolygon objects

ax = plt.axes(projection=ccrs.PlateCarree())

plt.title('France')
ax.coastlines(resolution='10m')

ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),
                  edgecolor='black', facecolor='gray', alpha=0.5)

ax.set_extent([-10, 16, 37, 56], ccrs.PlateCarree())

plt.show()
