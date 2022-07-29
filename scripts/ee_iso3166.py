import pandas as pd

df = pd.read_csv('../data/iso3166.csv')
countries = df['alpha-3'].tolist()
for c in countries:
    print(c)
#print(countries)