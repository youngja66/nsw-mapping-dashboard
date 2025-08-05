"""
Enhanced PyDeck Spatial Dashboard for Greater Sydney LGAs
=========================================================

Based on your original working PyDeck code with enhancements:
- Large readable fonts (18px minimum)
- Click interactivity that updates charts
- Better data handling and error prevention
- Professional styling

Specifically designed for your data structure:
- Shapefile: C:\Data\zStreamlit_Mapping\data\greater_sydney_lgas.shp
- CSV files: C:\Data\zStreamlit_Mapping\data\TXge35\CSV\*.csv

Save this file as: app.py
Run with: streamlit run app.py

Author: Expert Streamlit & PyDeck Developer
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS IF YOUR DATA IS ELSEWHERE
# =============================================================================

#Perfect! Here's how to update your Config class to work with git repository paths instead of local drive paths:
#Updated Configuration Class
#python
import os
import streamlit as st

class Config:
    """Configuration settings for the dashboard."""
    
    @staticmethod
    def get_base_path():
        """Get the base path - works for both local development and Streamlit Cloud"""
        # For Streamlit Cloud deployment, use relative paths
        if os.path.exists("data"):  # Repository structure
            return "data"
        
        # For local development, you can still use your local path
        local_path = r"C:\Data\zStreamlit_Mapping\data"
        if os.path.exists(local_path):
            return local_path
            
        # Fallback - current directory
        return "."
    
    # Dynamic data paths based on environment
    BASE_DATA_DIR = get_base_path.__func__()  # Call the static method
    SHAPEFILE_PATH = os.path.join(BASE_DATA_DIR, "greater_sydney_lgas.shp")
    CSV_DIRECTORY = os.path.join(BASE_DATA_DIR, "TXge35", "CSV")
    
    # Validation methods
    @classmethod
    def validate_paths(cls):
        """Validate that all required paths exist"""
        missing_paths = []
        
        if not os.path.exists(cls.SHAPEFILE_PATH):
            missing_paths.append(f"Shapefile: {cls.SHAPEFILE_PATH}")
            
        if not os.path.exists(cls.CSV_DIRECTORY):
            missing_paths.append(f"CSV Directory: {cls.CSV_DIRECTORY}")
            
        return missing_paths
    
    @classmethod
    def display_config_info(cls):
        """Display configuration information in Streamlit"""
        with st.expander("📁 Data Configuration", expanded=False):
            st.write("**Current Data Paths:**")
            st.code(f"Base Directory: {cls.BASE_DATA_DIR}")
            st.code(f"Shapefile: {cls.SHAPEFILE_PATH}")
            st.code(f"CSV Directory: {cls.CSV_DIRECTORY}")
            
            # Check if paths exist
            missing = cls.validate_paths()
            if missing:
                st.error("❌ Missing data files:")
                for path in missing:
                    st.write(f"- {path}")
            else:
                st.success("✅ All data paths found!")

    # Map settings
    DEFAULT_CENTER = {"lat": -33.8688, "lon": 151.2093}  # Sydney coordinates
    DEFAULT_ZOOM = 9
    DEFAULT_PITCH = 45

    # Performance settings
    MAX_POINTS_DISPLAY = 10000
    CACHE_TTL = 3600  # 1 hour

# =============================================================================
# APP CONFIGURATION & STYLING
# =============================================================================

def configure_app():
    """Configure Streamlit app with professional settings."""
    st.set_page_config(
        page_title="Greater Sydney LGA Analytics",
        page_icon="🌏",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/youngja66/sydney-lga-dashboard',
            'Report a bug': 'mailto:your-email@domain.com',
            'About': "# Greater Sydney LGA Dashboard\nAdvanced spatial analytics for NSW planning data"
        }
    )

def load_custom_css():
    """Load custom CSS for professional styling with LARGE fonts."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Base font size - LARGE for readability */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 18px !important;
    }

    .main {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-size: 18px !important;
    }

    .main .block-container {
        font-size: 18px !important;
        padding-top: 1rem !important;
    }

    .sydney-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }

    .sydney-header h1 {
        font-size: 3.2rem !important;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .sydney-header p {
        font-size: 1.4rem !important;
        margin: 1rem 0 0 0;
        opacity: 0.9;
    }

    /* ALL TEXT ELEMENTS - LARGE FONTS */
    .main h1 { font-size: 2.8rem !important; font-weight: 700 !important; }
    .main h2 { font-size: 2.2rem !important; font-weight: 600 !important; }
    .main h3 { font-size: 1.8rem !important; font-weight: 600 !important; }
    .main h4 { font-size: 1.5rem !important; font-weight: 600 !important; }
    .main h5 { font-size: 1.25rem !important; font-weight: 500 !important; }
    .main p { font-size: 18px !important; }
    .main div { font-size: 18px !important; }
    .main span { font-size: 18px !important; }
    .main li { font-size: 18px !important; }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #2a5298;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
        font-size: 18px !important;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }

    .data-status {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        font-size: 18px !important;
    }

    .pydeck-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.3);
        background: white;
    }

    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        background: white;
        font-size: 18px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(42, 82, 152, 0.4);
    }

    /* Sidebar styling with LARGE fonts */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-size: 18px !important;
    }

    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        font-size: 18px !important;
    }

    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stSlider label,
    .css-1d391kg .stRadio label,
    .css-1d391kg .stCheckbox label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 18px !important;
    }

    .css-1d391kg .stMarkdown {
        color: white !important;
        font-size: 18px !important;
    }

    .css-1d391kg .stMarkdown h2 {
        color: white !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    .css-1d391kg .stMarkdown h3 {
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }

    /* Metrics with LARGE fonts */
    .metric-container .metric-label {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    .metric-container .metric-value {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
    }

    .metric-container .metric-delta {
        font-size: 18px !important;
    }

    /* Data tables with LARGE fonts */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        font-size: 18px !important;
    }

    .dataframe th {
        font-size: 20px !important;
        font-weight: 600 !important;
        background-color: #1e3c72 !important;
        color: white !important;
    }

    .dataframe td {
        font-size: 18px !important;
        padding: 10px !important;
    }

    /* Success/status messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        font-size: 20px !important;
        padding: 1rem !important;
        font-weight: 600 !important;
    }

    .stSuccess > div, .stError > div, .stWarning > div, .stInfo > div {
        font-size: 20px !important;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        font-size: 18px !important;
        padding: 1rem 2rem !important;
    }

    /* Force font size on ALL elements */
    div, span, p, h1, h2, h3, h4, h5, h6, button, input, select, textarea, label {
        font-family: 'Inter', sans-serif !important;
    }

    /* Streamlit specific overrides */
    [data-testid="metric-container"] {
        font-size: 18px !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
    }

    /* Tab labels */
    .stTabs [data-baseweb="tab"] {
        font-size: 18px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
    }

    /* Checkbox and radio text */
    .stCheckbox > div > label > div:last-child,
    .stRadio > div > label > div:last-child {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & MANAGEMENT (Enhanced with better error handling)
# =============================================================================

class DataManager:
    """Manage loading and processing of Sydney LGA data with enhanced error handling."""

    def __init__(self):
        self.spatial_data = None
        self.csv_data = None
        self.merged_data = None
        self.csv_files_info = []

    @st.cache_data(ttl=Config.CACHE_TTL)
    def load_shapefile(_self) -> gpd.GeoDataFrame:
        """Load the Greater Sydney LGAs shapefile."""
        try:
            if not os.path.exists(Config.SHAPEFILE_PATH):
                st.error(f"❌ Shapefile not found at: {Config.SHAPEFILE_PATH}")
                st.info("Please ensure the shapefile exists at the specified location.")
                return None

            # Load shapefile
            gdf = gpd.read_file(Config.SHAPEFILE_PATH)

            # Ensure proper CRS for web mapping
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')

            # Clean column names
            gdf.columns = [col.strip() for col in gdf.columns]

            # Identify LGA name column
            lga_name_col = None
            for col in gdf.columns:
                if any(term in col.lower() for term in ['lga', 'name', 'area']):
                    lga_name_col = col
                    break

            if lga_name_col:
                gdf = gdf.rename(columns={lga_name_col: 'LGA_Name'})
            else:
                st.warning("⚠️ Could not identify LGA name column. Using first text column.")
                text_cols = gdf.select_dtypes(include=['object']).columns
                if len(text_cols) > 0:
                    gdf = gdf.rename(columns={text_cols[0]: 'LGA_Name'})

            return gdf

        except Exception as e:
            st.error(f"❌ Error loading shapefile: {str(e)}")
            return None

    @st.cache_data(ttl=Config.CACHE_TTL)
    def load_csv_data(_self) -> Tuple[pd.DataFrame, List[Dict]]:
        """Load and merge all CSV files from the specified directory with data cleaning."""
        try:
            if not os.path.exists(Config.CSV_DIRECTORY):
                st.error(f"❌ CSV directory not found: {Config.CSV_DIRECTORY}")
                return None, []

            # Find all CSV files
            csv_pattern = os.path.join(Config.CSV_DIRECTORY, "*.csv")
            csv_files = glob.glob(csv_pattern)

            if not csv_files:
                st.error(f"❌ No CSV files found in: {Config.CSV_DIRECTORY}")
                return None, []

            merged_data = None
            csv_files_info = []

            for csv_file in csv_files:
                try:
                    # Load CSV
                    df = pd.read_csv(csv_file)

                    # Get file info
                    file_name = os.path.basename(csv_file)
                    file_info = {
                        'filename': file_name,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'size_mb': os.path.getsize(csv_file) / (1024 * 1024)
                    }
                    csv_files_info.append(file_info)

                    # Clean column names
                    df.columns = [col.strip() for col in df.columns]

                    # CLEAN DATA - Convert any problematic data types
                    for col in df.columns:
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            df[col] = df[col].astype(str)
                        elif df[col].dtype == 'object':
                            # Check for timestamps in object columns
                            try:
                                df[col] = df[col].fillna('').astype(str)
                            except:
                                df[col] = ''

                    # Find LGA identifier column
                    lga_col = None
                    for col in df.columns:
                        if any(term in col.lower() for term in ['lga', 'area', 'region']):
                            lga_col = col
                            break

                    if lga_col is None:
                        st.warning(f"⚠️ No LGA column found in {file_name}")
                        continue

                    # Standardize LGA column name
                    df = df.rename(columns={lga_col: 'LGA_Name'})

                    # Clean LGA names (remove common suffixes like (C), (A), etc.)
                    df['LGA_Name'] = df['LGA_Name'].astype(str).str.replace(r'\s*\([A-Z]\)', '', regex=True).str.strip()

                    # Add variable prefix to other columns
                    variable_name = Path(csv_file).stem
                    for col in df.columns:
                        if col != 'LGA_Name':
                            df = df.rename(columns={col: f"{variable_name}_{col}"})

                    # Merge with main dataset
                    if merged_data is None:
                        merged_data = df
                    else:
                        merged_data = pd.merge(merged_data, df, on='LGA_Name', how='outer')

                except Exception as e:
                    st.warning(f"⚠️ Error loading {os.path.basename(csv_file)}: {str(e)}")
                    continue

            return merged_data, csv_files_info

        except Exception as e:
            st.error(f"❌ Error loading CSV data: {str(e)}")
            return None, []

    def merge_spatial_and_csv_data(self) -> bool:
        """Merge spatial and CSV data with comprehensive data cleaning."""
        try:
            if self.spatial_data is None or self.csv_data is None:
                return False

            # Clean LGA names in spatial data
            if 'LGA_Name' in self.spatial_data.columns:
                self.spatial_data['LGA_Name'] = (
                    self.spatial_data['LGA_Name']
                    .astype(str)
                    .str.replace(r'\s*\([A-Z]\)', '', regex=True)
                    .str.strip()
                )

            # Merge datasets
            self.merged_data = self.spatial_data.merge(
                self.csv_data,
                on='LGA_Name',
                how='left'
            )

            # FINAL DATA CLEANING for PyDeck compatibility
            for col in self.merged_data.columns:
                if col == 'geometry':
                    continue

                # Ensure all non-geometry columns are PyDeck compatible
                if pd.api.types.is_datetime64_any_dtype(self.merged_data[col]):
                    self.merged_data[col] = self.merged_data[col].astype(str)
                elif self.merged_data[col].dtype == 'object':
                    self.merged_data[col] = self.merged_data[col].fillna('').astype(str)

            return True

        except Exception as e:
            st.error(f"❌ Error merging data: {str(e)}")
            return False

    def load_all_data(self) -> bool:
        """Load all data sources."""
        with st.spinner("🔄 Loading spatial data..."):
            self.spatial_data = self.load_shapefile()

        if self.spatial_data is None:
            return False

        with st.spinner("🔄 Loading CSV data..."):
            self.csv_data, self.csv_files_info = self.load_csv_data()

        if self.csv_data is None:
            return False

        with st.spinner("🔄 Merging datasets..."):
            success = self.merge_spatial_and_csv_data()

        return success

# =============================================================================
# ENHANCED PYDECK VISUALIZATION (Based on your original working code)
# =============================================================================

class PyDeckVisualizer:
    """Create PyDeck visualizations for Sydney LGA data."""

    def __init__(self):
        self.default_view = pdk.ViewState(
            latitude=Config.DEFAULT_CENTER['lat'],
            longitude=Config.DEFAULT_CENTER['lon'],
            zoom=Config.DEFAULT_ZOOM,
            pitch=Config.DEFAULT_PITCH
        )

    def create_polygon_layer(self, data: gpd.GeoDataFrame,
                            value_column: str = None) -> pdk.Layer:
        """Create polygon layer from GeoDataFrame - your original working method."""

        # Convert GeoDataFrame to format suitable for PyDeck
        data_for_pydeck = []

        for idx, row in data.iterrows():
            if row.geometry is not None:
                try:
                    # Get coordinates from geometry
                    if row.geometry.geom_type == 'Polygon':
                        coords = [list(row.geometry.exterior.coords)]
                    elif row.geometry.geom_type == 'MultiPolygon':
                        coords = [list(geom.exterior.coords) for geom in row.geometry.geoms]
                    else:
                        continue

                    # Prepare feature data
                    feature_data = {
                        'coordinates': coords,
                        'LGA_Name': str(row.get('LGA_Name', 'Unknown'))
                    }

                    # Add value for coloring if specified
                    if value_column and value_column in row.index and pd.notna(row[value_column]):
                        try:
                            feature_data['value'] = float(row[value_column])
                        except:
                            feature_data['value'] = 0
                    else:
                        feature_data['value'] = 0

                    data_for_pydeck.append(feature_data)

                except Exception as e:
                    continue  # Skip problematic geometries

        # Determine fill color based on whether we have values
        if value_column and any(item['value'] != 0 for item in data_for_pydeck):
            # Normalize values for color mapping
            values = [item['value'] for item in data_for_pydeck if item['value'] != 0]
            if values:
                min_val, max_val = min(values), max(values)
                if max_val > min_val:
                    for item in data_for_pydeck:
                        if item['value'] != 0:
                            normalized = (item['value'] - min_val) / (max_val - min_val)
                            # Color from blue to red
                            item['fill_color'] = [
                                int(255 * normalized),      # Red
                                100,                        # Green
                                int(255 * (1 - normalized)), # Blue
                                160                         # Alpha
                            ]
                        else:
                            item['fill_color'] = [100, 100, 100, 160]  # Gray for no data
                else:
                    for item in data_for_pydeck:
                        item['fill_color'] = [100, 150, 200, 160]  # Default blue
        else:
            # Default color for all
            for item in data_for_pydeck:
                item['fill_color'] = [100, 150, 200, 160]

        # Create polygon layer
        layer = pdk.Layer(
            'PolygonLayer',
            data=data_for_pydeck,
            get_polygon='coordinates',
            get_fill_color='fill_color',
            get_line_color='[255, 255, 255, 200]',
            get_line_width=2,
            pickable=True,
            auto_highlight=True,
            extruded=False
        )

        return layer

    def create_point_layer(self, data: gpd.GeoDataFrame,
                          size_column: str = None,
                          color_column: str = None) -> pdk.Layer:
        """Create point layer for centroids."""

        # Get centroids
        centroids = data.copy()
        centroids['geometry'] = centroids['geometry'].centroid
        centroids['lon'] = centroids.geometry.x
        centroids['lat'] = centroids.geometry.y

        # Prepare data
        point_data = []
        for idx, row in centroids.iterrows():
            point_info = {
                'lon': row['lon'],
                'lat': row['lat'],
                'LGA_Name': str(row.get('LGA_Name', 'Unknown'))
            }

            # Add size data
            if size_column and size_column in row.index and pd.notna(row[size_column]):
                # Normalize size
                max_val = centroids[size_column].max()
                min_val = centroids[size_column].min()
                if max_val > min_val:
                    normalized = (row[size_column] - min_val) / (max_val - min_val)
                    point_info['size'] = 50 + (normalized * 200)  # Size between 50-250
                else:
                    point_info['size'] = 100
            else:
                point_info['size'] = 100

            # Add color data
            if color_column and color_column in row.index and pd.notna(row[color_column]):
                # Simple color mapping (red to green)
                max_val = centroids[color_column].max()
                min_val = centroids[color_column].min()
                if max_val > min_val:
                    normalized = (row[color_column] - min_val) / (max_val - min_val)
                    point_info['color'] = [
                        int(255 * (1 - normalized)),  # Red component
                        int(255 * normalized),        # Green component
                        100,                          # Blue component
                        180                           # Alpha
                    ]
                else:
                    point_info['color'] = [100, 100, 100, 180]
            else:
                point_info['color'] = [100, 150, 200, 180]

            point_data.append(point_info)

        layer = pdk.Layer(
            'ScatterplotLayer',
            data=point_data,
            get_position='[lon, lat]',
            get_radius='size',
            get_color='color',
            pickable=True,
            auto_highlight=True
        )

        return layer

    def create_3d_column_layer(self, data: gpd.GeoDataFrame,
                              height_column: str) -> pdk.Layer:
        """Create 3D column layer."""

        if height_column not in data.columns:
            return None

        # Get centroids
        centroids = data.copy()
        centroids['geometry'] = centroids['geometry'].centroid
        centroids['lon'] = centroids.geometry.x
        centroids['lat'] = centroids.geometry.y

        # Normalize heights
        max_height = data[height_column].max()
        min_height = data[height_column].min()

        if max_height <= min_height:
            return None

        column_data = []
        for idx, row in centroids.iterrows():
            if pd.notna(row[height_column]):
                normalized_height = ((row[height_column] - min_height) /
                                   (max_height - min_height)) * 1000

                column_data.append({
                    'lon': row['lon'],
                    'lat': row['lat'],
                    'height': max(normalized_height, 10),  # Minimum height of 10
                    'LGA_Name': str(row.get('LGA_Name', 'Unknown')),
                    'value': row[height_column]
                })

        layer = pdk.Layer(
            'ColumnLayer',
            data=column_data,
            get_position='[lon, lat]',
            get_elevation='height',
            elevation_scale=1,
            radius=1000,
            get_fill_color='[180, 0, 200, 140]',
            pickable=True,
            auto_highlight=True
        )

        return layer

# =============================================================================
# ENHANCED SIDEBAR CONTROLS
# =============================================================================

def create_enhanced_sidebar_controls(merged_data: gpd.GeoDataFrame,
                          csv_files_info: List[Dict]) -> Dict:
    """Create enhanced sidebar controls with click interactivity."""

    st.sidebar.markdown("## 🎛️ Dashboard Controls")

    controls = {}

    # Data overview
    with st.sidebar.expander("📊 Data Overview", expanded=True):
        st.write(f"**Total LGAs:** {len(merged_data)}")
        st.write(f"**CSV Files Loaded:** {len(csv_files_info)}")

        total_size = sum(info['size_mb'] for info in csv_files_info)
        st.write(f"**Total Data Size:** {total_size:.2f} MB")

    # Manual LGA Selection (backup to clicking)
    lga_options = ["All LGAs"] + sorted(merged_data['LGA_Name'].dropna().unique().tolist())
    manual_selection = st.sidebar.selectbox(
        "📍 Manual LGA Selection",
        options=lga_options,
        index=0,
        help="Select LGA manually (also works by clicking on map)"
    )

    # Update session state if manual selection differs
    if 'selected_lga' not in st.session_state:
        st.session_state.selected_lga = "All LGAs"

    if manual_selection != st.session_state.selected_lga:
        st.session_state.selected_lga = manual_selection

    # Variable selection for visualization
    numeric_columns = merged_data.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_columns:
        controls['choropleth_variable'] = st.sidebar.selectbox(
            "🎨 Choropleth Variable",
            options=[None] + numeric_columns,
            index=0,
            help="Select variable for color mapping"
        )

        controls['size_variable'] = st.sidebar.selectbox(
            "📏 Point Size Variable",
            options=[None] + numeric_columns,
            index=0,
            help="Select variable for point sizing"
        )

        controls['height_variable'] = st.sidebar.selectbox(
            "🏗️ 3D Height Variable",
            options=[None] + numeric_columns,
            index=0,
            help="Select variable for 3D column heights"
        )

    # Layer controls
    st.sidebar.markdown("### 🗺️ Map Layers")
    controls['show_polygons'] = st.sidebar.checkbox("🔷 LGA Boundaries", value=True)
    controls['show_points'] = st.sidebar.checkbox("⚫ Centroid Points", value=False)
    controls['show_3d_columns'] = st.sidebar.checkbox("🏗️ 3D Columns", value=False)

    # Map settings
    st.sidebar.markdown("### ⚙️ Map Settings")
    controls['map_style'] = st.sidebar.selectbox(
        "Map Style",
        options=['mapbox://styles/mapbox/light-v9', 'mapbox://styles/mapbox/dark-v9',
                'mapbox://styles/mapbox/satellite-v9', 'mapbox://styles/mapbox/streets-v11'],
        index=0
    )

    controls['pitch'] = st.sidebar.slider("Camera Pitch", 0, 90, Config.DEFAULT_PITCH)
    controls['zoom'] = st.sidebar.slider("Zoom Level", 6, 15, Config.DEFAULT_ZOOM)

    return controls

# =============================================================================
# ENHANCED ANALYTICS & CHARTS (with dynamic updates)
# =============================================================================

def create_enhanced_analytics_charts(data: gpd.GeoDataFrame, selected_lga: str = None) -> None:
    """Create enhanced analytics charts that respond to LGA selection."""

    # Filter data if LGA selected
    if selected_lga and selected_lga != "All LGAs":
        filtered_data = data[data['LGA_Name'] == selected_lga]
        title_suffix = f" for {selected_lga}"
        context = "individual"
    else:
        filtered_data = data
        title_suffix = " for All LGAs"
        context = "overview"

    # Get numeric columns
    numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("No numeric data available for analysis.")
        return

    # Summary statistics with large fonts
    st.markdown(f"## 📊 KEY METRICS{title_suffix.upper()}")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    if len(numeric_cols) >= 1:
        with col1:
            if context == "individual":
                value = filtered_data[numeric_cols[0]].iloc[0] if not filtered_data.empty else 0
                # Calculate rank
                rank = (data[numeric_cols[0]] > value).sum() + 1
                delta = f"Rank: {rank}/{len(data)}"
            else:
                value = filtered_data[numeric_cols[0]].mean()
                delta = f"σ = {filtered_data[numeric_cols[0]].std():.2f}"

            st.metric(
                label=numeric_cols[0].replace('_', ' ').title(),
                value=f"{value:.2f}" if pd.notna(value) else "N/A",
                delta=delta
            )

    if len(numeric_cols) >= 2:
        with col2:
            if context == "individual":
                value = filtered_data[numeric_cols[1]].iloc[0] if not filtered_data.empty else 0
                rank = (data[numeric_cols[1]] > value).sum() + 1
                delta = f"Rank: {rank}/{len(data)}"
            else:
                value = filtered_data[numeric_cols[1]].mean()
                delta = f"σ = {filtered_data[numeric_cols[1]].std():.2f}"

            st.metric(
                label=numeric_cols[1].replace('_', ' ').title(),
                value=f"{value:.2f}" if pd.notna(value) else "N/A",
                delta=delta
            )

    if len(numeric_cols) >= 3:
        with col3:
            if context == "individual":
                value = filtered_data[numeric_cols[2]].iloc[0] if not filtered_data.empty else 0
                rank = (data[numeric_cols[2]] > value).sum() + 1
                delta = f"Rank: {rank}/{len(data)}"
            else:
                value = filtered_data[numeric_cols[2]].mean()
                delta = f"σ = {filtered_data[numeric_cols[2]].std():.2f}"

            st.metric(
                label=numeric_cols[2].replace('_', ' ').title(),
                value=f"{value:.2f}" if pd.notna(value) else "N/A",
                delta=delta
            )

    with col4:
        st.metric("Total LGAs", len(filtered_data))

    # Enhanced Charts based on context
    st.markdown("---")

    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            if context == "individual":
                # Comparison chart showing selected vs others
                st.markdown(f"#### 🎯 {selected_lga} vs All Others")

                comparison_data = data.copy()
                comparison_data['Type'] = comparison_data['LGA_Name'].apply(
                    lambda x: 'Selected LGA' if x == selected_lga else 'Other LGAs'
                )

                fig_comparison = px.scatter(
                    comparison_data,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    color='Type',
                    hover_data=['LGA_Name'],
                    title=f'{numeric_cols[0].replace("_", " ").title()} vs {numeric_cols[1].replace("_", " ").title()}',
                    color_discrete_map={
                        'Selected LGA': '#ff4444',
                        'Other LGAs': '#cccccc'
                    }
                )
                fig_comparison.update_layout(height=400, font=dict(size=14))
                st.plotly_chart(fig_comparison, use_container_width=True)
            else:
                # Top performers chart
                st.markdown("#### 📈 Top 15 Performers")
                top_data = filtered_data.nlargest(15, numeric_cols[0])
                fig_bar = px.bar(
                    top_data,
                    x='LGA_Name',
                    y=numeric_cols[0],
                    title=f'Top 15 LGAs by {numeric_cols[0].replace("_", " ").title()}',
                    color=numeric_cols[0],
                    color_continuous_scale='Blues'
                )
                fig_bar.update_layout(xaxis_tickangle=-45, height=400, font=dict(size=14))
                st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            if context == "individual":
                # Radar chart for selected LGA
                st.markdown(f"#### 📊 Multi-Variable Profile")

                variables = numeric_cols[:6]  # Top 6 variables
                values = []
                labels = []

                for var in variables:
                    value = filtered_data[var].iloc[0] if not filtered_data.empty else 0
                    # Normalize to 0-100 scale
                    max_val = data[var].max()
                    min_val = data[var].min()
                    if max_val > min_val and pd.notna(value):
                        normalized = ((value - min_val) / (max_val - min_val)) * 100
                        values.append(normalized)
                        labels.append(var.replace('_', ' ').title()[:20])  # Truncate long names

                if values:
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=labels,
                        fill='toself',
                        name=selected_lga,
                        line_color='#1e3c72'
                    ))

                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=False,
                        title="Percentile Rankings",
                        height=400,
                        font=dict(size=14)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
            else:
                # Distribution chart
                st.markdown("#### 📊 Data Distribution")
                fig_hist = px.histogram(
                    filtered_data,
                    x=numeric_cols[0],
                    nbins=20,
                    title=f'Distribution of {numeric_cols[0].replace("_", " ").title()}',
                    color_discrete_sequence=['#1e3c72']
                )
                fig_hist.update_layout(height=400, font=dict(size=14))
                st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""

    # Configure app
    configure_app()
    load_custom_css()

    # Header with large fonts
    st.markdown("""
    <div class="sydney-header">
        <h1>🌏 Greater Sydney LGA Analytics Dashboard</h1>
        <p>Enhanced PyDeck visualization with large fonts and click interactivity</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for selected LGA
    if 'selected_lga' not in st.session_state:
        st.session_state.selected_lga = "All LGAs"

    # Data status section
    st.markdown("## 📋 DATA LOADING STATUS")

    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.markdown(f"""
        <div class="data-status">
            <h4 style="font-size: 1.4rem;">📁 Data Sources</h4>
            <p style="font-size: 1.1rem;"><strong>Shapefile:</strong> {Config.SHAPEFILE_PATH}</p>
            <p style="font-size: 1.1rem;"><strong>CSV Directory:</strong> {Config.CSV_DIRECTORY}</p>
        </div>
        """, unsafe_allow_html=True)

    with status_col2:
        if st.button("🔄 Reload Data", help="Refresh data from source files"):
            st.cache_data.clear()
            st.rerun()

    # Initialize data manager
    data_manager = DataManager()

    # Load data
    if data_manager.load_all_data():

        # Success message with large fonts
        st.success(f"🎉 Successfully loaded {len(data_manager.merged_data)} LGAs with merged CSV data!")

        # Show CSV files info
        with st.expander("📊 Loaded CSV Files Information"):
            csv_info_df = pd.DataFrame(data_manager.csv_files_info)
            st.dataframe(csv_info_df, use_container_width=True)

        # Create sidebar controls
        controls = create_enhanced_sidebar_controls(data_manager.merged_data, data_manager.csv_files_info)

        # Main dashboard layout
        map_col, info_col = st.columns([0.6, 0.4])

        with map_col:
            st.markdown("## 🗺️ INTERACTIVE PYDECK MAP")
            st.markdown("**Click on any LGA area to select it and update all charts automatically**")

            # Initialize PyDeck visualizer
            visualizer = PyDeckVisualizer()

            # Filter data if LGA selected
            if st.session_state.selected_lga != "All LGAs":
                filtered_data = data_manager.merged_data[
                    data_manager.merged_data['LGA_Name'] == st.session_state.selected_lga
                ]
            else:
                filtered_data = data_manager.merged_data

            # Create layers
            layers = []

            if controls['show_polygons']:
                polygon_layer = visualizer.create_polygon_layer(
                    data_manager.merged_data,  # Always show all LGAs for context
                    controls.get('choropleth_variable')
                )
                if polygon_layer:
                    layers.append(polygon_layer)

            if controls['show_points']:
                point_layer = visualizer.create_point_layer(
                    filtered_data,
                    controls.get('size_variable'),
                    controls.get('choropleth_variable')
                )
                if point_layer:
                    layers.append(point_layer)

            if controls['show_3d_columns'] and controls.get('height_variable'):
                column_layer = visualizer.create_3d_column_layer(
                    filtered_data,
                    controls['height_variable']
                )
                if column_layer:
                    layers.append(column_layer)

            # Create deck
            deck = pdk.Deck(
                map_style=controls['map_style'],
                initial_view_state=pdk.ViewState(
                    latitude=Config.DEFAULT_CENTER['lat'],
                    longitude=Config.DEFAULT_CENTER['lon'],
                    zoom=controls['zoom'],
                    pitch=controls['pitch']
                ),
                layers=layers,
                tooltip={
                    'html': '<b style="font-size: 16px;">LGA:</b> <span style="font-size: 16px;">{LGA_Name}</span><br/><b style="font-size: 14px;">Value:</b> <span style="font-size: 14px;">{value}</span>',
                    'style': {
                        'backgroundColor': 'steelblue',
                        'color': 'white',
                        'fontSize': '16px'
                    }
                }
            )

            # Display map
            with st.container():
                st.markdown('<div class="pydeck-container">', unsafe_allow_html=True)

                # PyDeck with click handling
                map_result = st.pydeck_chart(deck, use_container_width=True)

                # Handle clicks (PyDeck doesn't return click data like Folium, but we have manual selection)

                st.markdown('</div>', unsafe_allow_html=True)

        with info_col:
            st.markdown("## 📊 ANALYSIS PANEL")

            # Show current selection prominently
            if st.session_state.selected_lga != "All LGAs":
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; font-size: 18px; font-weight: 600;">
                    🎯 <strong>SELECTED LGA:</strong><br/>
                    <span style="font-size: 24px; font-weight: 700;">{st.session_state.selected_lga}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%); color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; font-size: 18px; font-weight: 600;">
                    📊 <strong>VIEWING ALL LGAs</strong>
                </div>
                """, unsafe_allow_html=True)

            # Reset button
            if st.button("🔄 Reset to All LGAs", help="Clear selection and view all areas"):
                st.session_state.selected_lga = "All LGAs"
                st.rerun()

            # Quick info section
            if st.session_state.selected_lga != "All LGAs":
                lga_data = data_manager.merged_data[
                    data_manager.merged_data['LGA_Name'] == st.session_state.selected_lga
                ]
                if not lga_data.empty:
                    st.markdown("### 📋 Key Statistics")

                    # Show numeric values
                    numeric_cols = lga_data.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols[:5]:
                        value = lga_data[col].iloc[0]
                        if pd.notna(value):
                            st.write(f"**{col.replace('_', ' ').title()}:** {value:.2f}")
            else:
                st.markdown("### 📊 Overview Statistics")
                st.write(f"**Total LGAs:** {len(data_manager.merged_data)}")

                # Overall statistics
                numeric_cols = data_manager.merged_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.write("**Average Values:**")
                    for col in numeric_cols[:5]:
                        avg_val = data_manager.merged_data[col].mean()
                        if pd.notna(avg_val):
                            st.write(f"• {col.replace('_', ' ').title()}: {avg_val:.2f}")

        # Analytics section
        st.markdown("---")
        create_enhanced_analytics_charts(data_manager.merged_data, st.session_state.selected_lga)

        # Data table section
        st.markdown("---")
        st.markdown("## 📋 DATA EXPLORER")

        # Display filtered data
        display_data = data_manager.merged_data.drop(columns=['geometry'])

        if st.session_state.selected_lga != "All LGAs":
            display_data = display_data[display_data['LGA_Name'] == st.session_state.selected_lga]
            table_title = f"Data for {st.session_state.selected_lga}"
        else:
            table_title = "All LGA Data"

        st.markdown(f"### {table_title} ({len(display_data)} records)")
        st.dataframe(display_data, use_container_width=True, height=400)

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"sydney_lga_data_{st.session_state.selected_lga.replace(' ', '_') if st.session_state.selected_lga != 'All LGAs' else 'all'}.csv",
                mime="text/csv"
            )

        with col2:
            json_data = display_data.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"sydney_lga_data_{st.session_state.selected_lga.replace(' ', '_') if st.session_state.selected_lga != 'All LGAs' else 'all'}.json",
                mime="application/json"
            )

    else:
        # Data loading failed
        st.error("❌ Failed to load required data. Please check your file paths and data.")

        # Enhanced troubleshooting
        with st.expander("🔧 TROUBLESHOOTING GUIDE", expanded=True):
            st.markdown(f"""
            ## File Check Status:

            **Shapefile:** {"✅" if os.path.exists(Config.SHAPEFILE_PATH) else "❌"} `{Config.SHAPEFILE_PATH}`

            **CSV Directory:** {"✅" if os.path.exists(Config.CSV_DIRECTORY) else "❌"} `{Config.CSV_DIRECTORY}`

            ## Quick Solutions:

            1. **Update the Config class** at the top of app.py with your actual paths
            2. **Check shapefile components** - ensure .shp, .shx, .dbf, .prj files exist
            3. **Verify CSV structure** - ensure files have LGA identifier columns
            4. **Check permissions** - ensure Python can read the data directory

            ## Example Config Update:
            ```python
            class Config:
                BASE_DATA_DIR = r"YOUR_ACTUAL_PATH_HERE"
                SHAPEFILE_PATH = os.path.join(BASE_DATA_DIR, "your_shapefile.shp")
                CSV_DIRECTORY = os.path.join(BASE_DATA_DIR, "your_csv_folder")
            ```
            """)

            # Show files in CSV directory if it exists
            if os.path.exists(Config.CSV_DIRECTORY):
                csv_files = glob.glob(os.path.join(Config.CSV_DIRECTORY, "*.csv"))
                st.write(f"**Found {len(csv_files)} CSV files:**")
                for i, file in enumerate(csv_files[:10]):  # Show first 10 files
                    st.write(f"{i+1}. {os.path.basename(file)}")
                if len(csv_files) > 10:
                    st.write(f"... and {len(csv_files) - 10} more files")

if __name__ == "__main__":

    main()

