import folium
from PIL import Image
import io, base64

img = Image.open('scratch/subset_reprojected.tif')

m = folium.Map([37, 0], zoom_start=1)


b = io.BytesIO()
img.save(b, format='PNG')
b64 = base64.b64encode(b.getvalue())
folium.raster_layers.ImageOverlay(
    image=f'data:image/png;base64,{ b64.decode("utf-8") }',
    name=f'my overlay',
    bounds=[[-82, -180], [82, 180]],
    opacity=1,
    interactive=False,
    cross_origin=False,
    zindex=1,
    alt='Wikipedia File:Mercator projection SW.jpg'
).add_to(m)

folium.LayerControl().add_to(m)

m.save('scratch/map.html')