import geopandas as gpd
import pandas as pd
import os
import glob

# Test shapefile
try:
    shapefile_path = r"C:\Data\zStreamlit_Mapping\data\greater_sydney_lgas.shp"
    gdf = gpd.read_file(shapefile_path)
    print(f"✅ Shapefile: {len(gdf)} features loaded")
    print(f"Columns: {list(gdf.columns)}")
except Exception as e:
    print(f"❌ Shapefile error: {e}")

# Test CSV files
try:
    csv_directory = r"C:\Data\zStreamlit_Mapping\data\TXge35\CSV"
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    print(f"✅ Found {len(csv_files)} CSV files")
    for csv_file in csv_files[:3]:
        df = pd.read_csv(csv_file)
        print(f"  - {os.path.basename(csv_file)}: {len(df)} rows")
except Exception as e:
    print(f"❌ CSV error: {e}")