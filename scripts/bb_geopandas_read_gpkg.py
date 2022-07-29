import pandas as pd
import geopandas as gpd

if __name__ == '__main__':
    countries = set(pd.read_csv('../data/iso3166.csv')['alpha-3'].tolist())

    # [348904 rows x 50 columns] - takes a looong time
    df = gpd.read_file('../data/gadm_410.gpkg', rows=1000)
    gpkg_countries = set(df.GID_0.unique().tolist())

    print([x for x in gpkg_countries if x not in countries])