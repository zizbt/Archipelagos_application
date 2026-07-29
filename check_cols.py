import geopandas as gpd
gdf = gpd.read_file('data/gis/wdpa.geojson')
print(gdf.columns.tolist())
print(gdf[['NAME']].head(3))