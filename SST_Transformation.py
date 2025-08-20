import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from matplotlib import gridspec

# Set font to Times New Roman for all plots
plt.rcParams['font.family'] = 'Times New Roman'

def compute_spatial_std(data, radius_deg=2):
    """
    Compute the spatial standard deviation of SST within a specified radius for each point.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'Latitude', 'Longitude', and 'sst' columns.
    - radius_deg (float): Radius in degrees to search for neighboring points.

    Returns:
    - np.ndarray: Array of standard deviation values for each point.
    """
    coords = data[['Latitude', 'Longitude']].values
    tree = cKDTree(coords)
    std_sst = np.empty(len(data))
    std_sst.fill(np.nan)

    # Iterate over each point to find neighbors and compute std deviation
    for idx, point in enumerate(coords):
        indices = tree.query_ball_point(point, r=radius_deg)
        if len(indices) > 1:  # At least two points to compute std deviation
            std_sst[idx] = data.iloc[indices]['sst'].std(ddof=1)
    return std_sst

def exclude_area(data, lon_min_excl, lon_max_excl, lat_min_excl, lat_max_excl):
    """
    Exclude data points within a specified rectangular area.

    Parameters:
    - data (pd.DataFrame): Original DataFrame with 'Longitude' and 'Latitude' columns.
    - lon_min_excl (float): Minimum Longitude of exclusion zone.
    - lon_max_excl (float): Maximum Longitude of exclusion zone.
    - lat_min_excl (float): Minimum Latitude of exclusion zone.
    - lat_max_excl (float): Maximum Latitude of exclusion zone.

    Returns:
    - pd.DataFrame: DataFrame after excluding specified area.
    """
    # Create a boolean mask for points outside the exclusion zone
    mask = ~(
        (data['Longitude'] >= lon_min_excl) & (data['Longitude'] <= lon_max_excl) & 
        (data['Latitude'] >= lat_min_excl) & (data['Latitude'] <= lat_max_excl)
    )
    excluded_count = len(data) - mask.sum()
    print(f"Excluded {excluded_count} points within the rectangular exclusion area (Longitude: {lon_min_excl}-{lon_max_excl}, Latitude: {lat_min_excl}-{lat_max_excl}).")
    return data[mask]

def exclude_lat_below(data, lat_threshold=22):
    """
    Exclude data points with Latitude below a specified threshold.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'Latitude' column.
    - lat_threshold (float): Latitude threshold below which data points are excluded.

    Returns:
    - pd.DataFrame: DataFrame after excluding points with Latitude below the threshold.
    """
    mask = data['Latitude'] >= lat_threshold
    excluded_count = len(data) - mask.sum()
    print(f"Excluded {excluded_count} points with Latitude below {lat_threshold}°.")
    return data[mask]

def main():
    # Input file path
    file_path = r'C:\Users\DFMRendering\Desktop\Oman climate week\Visualization\Maybe_Final\Spatial_SST\AQUA_MODIS.20210601_20210630.L3m.MO.SST.x_sst_Best_Mask.txt'

    # Read the data
    try:
        data = pd.read_csv(file_path, delim_whitespace=True)
        print("Data successfully loaded.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Check for required columns
    required_columns = {'Latitude', 'Longitude', 'sst'}
    if not required_columns.issubset(data.columns):
        print(f"Input data must contain the following columns: {required_columns}")
        return

    # Drop rows with NaN in 'sst' column
    initial_count = len(data)
    data = data.dropna(subset=['sst'])
    dropped_count = initial_count - len(data)
    print(f"Dropped {dropped_count} rows with NaN in 'sst' column.")

    # Define exclusion zone boundaries
    lon_min_excl = 66
    lon_max_excl = 70
    lat_min_excl = 26
    lat_max_excl = 28
    lat_threshold = 23  # Latitude threshold for exclusion

    # Exclude data points within the specified rectangular area
    data = exclude_area(data, lon_min_excl, lon_max_excl, lat_min_excl, lat_max_excl)

    # Exclude data points with Latitude below the threshold
    data = exclude_lat_below(data, lat_threshold=lat_threshold)

    # Check if data is empty after exclusions
    if data.empty:
        print("No data points remain after applying the exclusion zones. Exiting script.")
        return

    # Set radius (in degrees)
    radius_deg = 1  # Approximately 200 km

    # Compute spatial standard deviation
    print("Calculating spatial standard deviation...")
    data['std_sst'] = compute_spatial_std(data, radius_deg=radius_deg)
    print("Spatial standard deviation calculation completed.")

    # Save the output CSV file
    output_file = os.path.splitext(file_path)[0] + '_SST_STD_Output.csv'
    try:
        data.to_csv(output_file, index=False, float_format='%.6f')  # Reduced precision for smaller file size
        print(f"Processed data saved to CSV file: {output_file}")
    except Exception as e:
        print(f"An error occurred while saving the CSV file: {e}")

    # Create 2D Heatmap using Cartopy with Horizontal Colorbar Below
    heatmap_file_2d = os.path.splitext(file_path)[0] + '_Heatmap_Cartopy.png'

    # Initialize the figure with GridSpec for precise layout control
    plt.figure(figsize=(12, 8), facecolor='white')  # Set figure background to white
    gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0)  # Reduce vertical spacing

    # Define the projection
    projection = ccrs.PlateCarree()

    # Create the main plot axes (first row)
    ax = plt.subplot(gs[0], projection=projection)
    ax.set_facecolor('white')  # Set axes background to white

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='white')  # Set land color to white
    ax.add_feature(cfeature.OCEAN, facecolor='white')  # Set ocean color to white
    ax.add_feature(cfeature.LAKES, facecolor='white')  # Set lakes color to white
    ax.add_feature(cfeature.RIVERS, edgecolor='blue')  # Set rivers color to blue

    # Set map extent based on data with some padding, ensuring no area below 23°N is included
    buffer = 1  # degrees
    min_lon, max_lon = data['Longitude'].min() - buffer, data['Longitude'].max() + buffer
    calculated_min_lat = data['Latitude'].min() - buffer
    min_lat = max(23, calculated_min_lat)
    max_lat = data['Latitude'].max() + buffer
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=projection)

    # Plot the data
    scatter = ax.scatter(
        data['Longitude'],
        data['Latitude'],
        c=data['std_sst'],
        cmap='Spectral',
        s=15,
        alpha=0.9,
        edgecolors='none',
        transform=projection
    )

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    # Add title
    plt.suptitle('2D Heatmap of SST Standard Deviation', fontsize=16, y=0.85)  # Adjust y-position to reduce spacing

    # Create the colorbar axes (second row)
    cbar_ax = plt.subplot(gs[1])

    # Add horizontal colorbar
    cbar = plt.colorbar(
        scatter,
        cax=cbar_ax,
        orientation='horizontal',
        fraction=1.0,
        pad=0.0
    )

    # Set colorbar label with increased font size and bold
    cbar.set_label('SST Standard Deviation (°C)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)  # Keep tick label size as is

    # Adjust layout to ensure the colorbar fits well
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout boundaries

    # Save the heatmap
    plt.savefig(heatmap_file_2d, dpi=400, bbox_inches='tight', facecolor='white')  # Ensure white background
    plt.close()

    print(f"2D Heatmap with horizontal colorbar saved to: {heatmap_file_2d}")

if __name__ == "__main__":
    main()