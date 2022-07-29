import fiona

# No need to pass "layer='etc'" if there's only one layer
# layer corresponds to <table_name> in gpkg_contents
with fiona.open('../data/gadm_410.gpkg', layer='gadm_410') as layer:
    i = 0
    for feature in layer:
        print(feature['geometry'])
        i += 1
        if i==10:
            break