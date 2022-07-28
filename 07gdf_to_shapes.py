import io
import base64
import geopandas
import folium
from folium import plugins
import rasterio
import rasterio.features
from PIL import Image


GDB = '/data/projects/kreike/data/KreikeSampleExtractedDataNam52022.gdb/'

if __name__ == '__main__':

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    # Set index to the country name (unique)
    world = world.set_index('name', drop=False)

    africa = world[(world['continent'] == 'Africa')]

    series = africa['geometry']  # GeoSeries
    for country, x in series.iteritems():
        # x is a shapely object
        pass

    geo_json1 = africa.geometry.to_json()
    geo_j1 = folium.GeoJson(data=geo_json1, style_function=lambda x: {'fillColor': 'orange'})

    gdf = geopandas.read_file(GDB, layer='Omuti1972')
    # Folium by default accepts lat/long (crs 4326) as input
    geo_json2 = gdf.to_crs(epsg=4326).geometry.to_json()
    geo_j2 = folium.GeoJson(data=geo_json2, style_function=lambda x: {'fillColor': 'orange'})

    raster_bounds = None
    with rasterio.open('subset_reprojected.tif') as raster:
        b = raster.bounds  # left, bottom, right, top
        raster_bounds = [[b[1], b[0]], [b[3], b[2]]]  # [[lat_min, lon_min], [lat_max, lon_max]]

    img = Image.open('subset_reprojected.tif')
    b = io.BytesIO()
    img.save(b, format='PNG')
    b64 = base64.b64encode(b.getvalue())

    map = folium.Map()
    geo_j1.add_to(map)
    geo_j2.add_to(map)

    overlay = folium.raster_layers.ImageOverlay(
        name='Aerial Image',
        image=f'data:image/png;base64,{ b64.decode("utf-8") }',
        bounds=raster_bounds,
        interactive=True,
        cross_origin=False,
        zindex=2,
    )
    overlay.add_to(map)
    folium.LayerControl().add_to(map)

    map.save('map.html')