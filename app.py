"""
Complete Custom PyDeck Spatial Dashboard for Greater Sydney LGAs
================================================================

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

class Config:
    """Configuration settings for the dashboard."""

    # Data paths - MODIFY THESE TO MATCH YOUR SETUP
    BASE_DATA_DIR = r"C:\Data\zStreamlit_Mapping\data"
    SHAPEFILE_PATH = os.path.join(BASE_DATA_DIR, "greater_sydney_lgas.shp")
    CSV_DIRECTORY = os.path.join(BASE_DATA_DIR, "TXge35", "CSV")

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
    """Load custom CSS for professional styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
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
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .sydney-header p {
        font-size: 2.2rem;
        margin: 1rem 0 0 0;
        opacity: 0.9;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #2a5298;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
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
    }

    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(42, 82, 152, 0.4);
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }

    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stSlider label,
    .css-1d391kg .stRadio label {
        color: white !important;
        font-weight: 500;
    }

    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & MANAGEMENT
# =============================================================================

class DataManager:
    """Manage loading and processing of Sydney LGA data."""

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
        """Load and merge all CSV files from the specified directory."""
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
        """Merge spatial and CSV data."""
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
# PYDECK VISUALIZATION
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
        """Create polygon layer from GeoDataFrame."""

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
                        'LGA_Name': row.get('LGA_Name', 'Unknown')
                    }

                    # Add value for coloring if specified
                    if value_column and value_column in row.index and pd.notna(row[value_column]):
                        feature_data['value'] = float(row[value_column])
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
                'LGA_Name': row.get('LGA_Name', 'Unknown')
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
                    'LGA_Name': row.get('LGA_Name', 'Unknown'),
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
# SIDEBAR CONTROLS
# =============================================================================

def create_sidebar_controls(merged_data: gpd.GeoDataFrame,
                          csv_files_info: List[Dict]) -> Dict:
    """Create sidebar controls for the dashboard."""

    st.sidebar.markdown("## 🎛️ Dashboard Controls")

    controls = {}

    # Data overview
    with st.sidebar.expander("📊 Data Overview", expanded=True):
        st.write(f"**Total LGAs:** {len(merged_data)}")
        st.write(f"**CSV Files Loaded:** {len(csv_files_info)}")

        total_size = sum(info['size_mb'] for info in csv_files_info)
        st.write(f"**Total Data Size:** {total_size:.2f} MB")

    # LGA Selection
    lga_options = ["All LGAs"] + sorted(merged_data['LGA_Name'].dropna().unique().tolist())
    controls['selected_lga'] = st.sidebar.selectbox(
        "📍 Select LGA",
        options=lga_options,
        index=0,
        help="Choose a specific LGA or view all"
    )

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
# ANALYTICS & CHARTS
# =============================================================================

def create_analytics_charts(data: gpd.GeoDataFrame, selected_lga: str = None) -> None:
    """Create analytics charts and summaries."""

    # Filter data if LGA selected
    if selected_lga and selected_lga != "All LGAs":
        filtered_data = data[data['LGA_Name'] == selected_lga]
        title_suffix = f" - {selected_lga}"
    else:
        filtered_data = data
        title_suffix = " - All LGAs"

    # Get numeric columns
    numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("No numeric data available for analysis.")
        return

    # Summary statistics
    st.markdown(f"### 📊 Summary Statistics{title_suffix}")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    if len(numeric_cols) >= 1:
        with col1:
            avg_val = filtered_data[numeric_cols[0]].mean()
            st.metric(
                label=numeric_cols[0].replace('_', ' ').title(),
                value=f"{avg_val:.2f}" if pd.notna(avg_val) else "N/A"
            )

    if len(numeric_cols) >= 2:
        with col2:
            avg_val = filtered_data[numeric_cols[1]].mean()
            st.metric(
                label=numeric_cols[1].replace('_', ' ').title(),
                value=f"{avg_val:.2f}" if pd.notna(avg_val) else "N/A"
            )

    if len(numeric_cols) >= 3:
        with col3:
            avg_val = filtered_data[numeric_cols[2]].mean()
            st.metric(
                label=numeric_cols[2].replace('_', ' ').title(),
                value=f"{avg_val:.2f}" if pd.notna(avg_val) else "N/A"
            )

    with col4:
        st.metric("Total LGAs", len(filtered_data))

    # Charts
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart of top LGAs
            top_data = filtered_data.nlargest(10, numeric_cols[0])
            fig_bar = px.bar(
                top_data,
                x='LGA_Name',
                y=numeric_cols[0],
                title=f'Top 10 LGAs by {numeric_cols[0].replace("_", " ").title()}',
                color=numeric_cols[0],
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Scatter plot
            fig_scatter = px.scatter(
                filtered_data,
                x=numeric_cols[0],
                y=numeric_cols[1],
                hover_data=['LGA_Name'],
                title=f'{numeric_cols[0].replace("_", " ").title()} vs {numeric_cols[1].replace("_", " ").title()}',
                color=numeric_cols[1] if len(numeric_cols) > 2 else None,
                color_continuous_scale='Viridis'
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""

    # Configure app
    configure_app()
    load_custom_css()

    # Header
    st.markdown("""
    <div class="sydney-header">
        <h1>🌏 Greater Sydney LGA Analytics Dashboard</h1>
        <p>Advanced spatial analysis of NSW planning data with PyDeck visualization</p>
    </div>
    """, unsafe_allow_html=True)

    # Data status section
    st.markdown("### 📋 Data Loading Status")

    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.markdown(f"""
        <div class="data-status">
            <h4>📁 Data Sources</h4>
            <p><strong>Shapefile:</strong> {Config.SHAPEFILE_PATH}</p>
            <p><strong>CSV Directory:</strong> {Config.CSV_DIRECTORY}</p>
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

        # Success message
        st.success(f"✅ Successfully loaded {len(data_manager.merged_data)} LGAs with merged CSV data!")

        # Show CSV files info
        with st.expander("📊 Loaded CSV Files Information"):
            csv_info_df = pd.DataFrame(data_manager.csv_files_info)
            st.dataframe(csv_info_df, use_container_width=True)

        # Create sidebar controls
        controls = create_sidebar_controls(data_manager.merged_data, data_manager.csv_files_info)

        # Main dashboard layout
        map_col, info_col = st.columns([0.7, 0.3])

        with map_col:
            st.markdown("### 🗺️ Interactive PyDeck Map")

            # Initialize PyDeck visualizer
            visualizer = PyDeckVisualizer()

            # Filter data if LGA selected
            if controls['selected_lga'] != "All LGAs":
                filtered_data = data_manager.merged_data[
                    data_manager.merged_data['LGA_Name'] == controls['selected_lga']
                ]
            else:
                filtered_data = data_manager.merged_data

            # Create layers
            layers = []

            if controls['show_polygons']:
                polygon_layer = visualizer.create_polygon_layer(
                    filtered_data,
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
                    'html': '<b>LGA:</b> {LGA_Name}<br/><b>Value:</b> {value}',
                    'style': {
                        'backgroundColor': 'steelblue',
                        'color': 'white'
                    }
                }
            )

            # Display map
            with st.container():
                st.markdown('<div class="pydeck-container">', unsafe_allow_html=True)
                st.pydeck_chart(deck, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with info_col:
            st.markdown("### 📊 Quick Info")

            # LGA info
            if controls['selected_lga'] != "All LGAs":
                lga_data = data_manager.merged_data[
                    data_manager.merged_data['LGA_Name'] == controls['selected_lga']
                ]
                if not lga_data.empty:
                    st.markdown(f"**Selected LGA:** {controls['selected_lga']}")

                    # Show numeric values
                    numeric_cols = lga_data.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols[:5]:
                        value = lga_data[col].iloc[0]
                        if pd.notna(value):
                            st.write(f"**{col.replace('_', ' ').title()}:** {value:.2f}")
            else:
                st.markdown("**Viewing:** All Greater Sydney LGAs")
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
        create_analytics_charts(data_manager.merged_data, controls['selected_lga'])

        # Data table section
        st.markdown("---")
        st.markdown("### 📋 Data Table")

        # Display filtered data
        display_data = data_manager.merged_data.drop(columns=['geometry'])

        if controls['selected_lga'] != "All LGAs":
            display_data = display_data[display_data['LGA_Name'] == controls['selected_lga']]

        st.dataframe(display_data, use_container_width=True, height=400)

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"sydney_lga_data_{controls['selected_lga'].replace(' ', '_') if controls['selected_lga'] != 'All LGAs' else 'all'}.csv",
                mime="text/csv"
            )

        with col2:
            json_data = display_data.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"sydney_lga_data_{controls['selected_lga'].replace(' ', '_') if controls['selected_lga'] != 'All LGAs' else 'all'}.json",
                mime="application/json"
            )

    else:
        # Data loading failed
        st.error("❌ Failed to load required data. Please check your file paths and data.")

        # Show troubleshooting info
        with st.expander("🔧 Troubleshooting"):
            st.markdown(f"""
            **Common Issues:**

            1. **File paths are incorrect**: Verify the paths in the Config class match your actual file locations
            2. **Files don't exist**: Ensure both shapefile and CSV files are in the specified locations
            3. **Permission issues**: Make sure Python has read access to the data directory
            4. **Column naming**: Ensure CSV files have an LGA identifier column

            **Current Configuration:**
            - Shapefile: `{Config.SHAPEFILE_PATH}`
            - CSV Directory: `{Config.CSV_DIRECTORY}`

            **File Existence Check:**
            - Shapefile exists: {"✅" if os.path.exists(Config.SHAPEFILE_PATH) else "❌"}
            - CSV directory exists: {"✅" if os.path.exists(Config.CSV_DIRECTORY) else "❌"}
            """)

            # Show files in CSV directory if it exists
            if os.path.exists(Config.CSV_DIRECTORY):
                csv_files = glob.glob(os.path.join(Config.CSV_DIRECTORY, "*.csv"))
                st.write(f"**CSV files found:** {len(csv_files)}")
                for file in csv_files[:5]:  # Show first 5 files
                    st.write(f"• {os.path.basename(file)}")

if __name__ == "__main__":
    main()