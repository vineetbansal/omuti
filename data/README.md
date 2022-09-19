internal data
sudo mount -t cifs //lockhart.princeton.edu/AfricaAirPhotos/Kreike /mnt/tmp -o username=vineetb@princeton.edu,uid=$(id -u)

.gpkg file
Something that geopandas can read - has metadata info and geometry information
Download the whole world data as a single .gpkg file from:

https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-gpkg.zip

France .shp file (and other files needed by a shapefile)
https://gadm.org/download_country.html

.tfw file
The X and Y pixel size, rotational information, and the world coordinates for the top-left corner of the TIF image.

.ovr file
pyramid file for ArcGIS - to allow fast rendering of map at various zoom levels.

.aux.xml
ArcGIS specific. Calculated statistics etc.

.xml
standard XML Files and widely adopted by many systems to store and read metadata that may or may not be in the header
of the TIF file
