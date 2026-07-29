import geopandas as gpd
gdf = gpd.read_file('data/gis/wdpa.geojson')
print('Colonnes:', gdf.columns.tolist())
print('Nb zones:', len(gdf))
print(gdf.head(10).to_string())