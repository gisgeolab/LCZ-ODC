
import io
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rasterio
from rasterio import mask
from rasterio.plot import show 
import numpy as np 
import leafmap 
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd 
from rasterio.mask import mask 
import ipywidgets as widgets
from IPython.display import display
from osgeo import gdal, osr
import pandas as pd 
from shapely.geometry import Polygon
import datacube
import sys
from pyproj import Transformer
import math
import math
import folium
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import colors as mcolours
from matplotlib.animation import FuncAnimation
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import box
from skimage.exposure import rescale_intensity
from tqdm.auto import tqdm
import rioxarray
import xarray as xr
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import shape
import geopandas as gdf
import matplotlib.pylab as pl
from matplotlib import gridspec
from ipywidgets import interactive
from PIL import Image
import json
from xrspatial import zonal_stats
import seaborn as sns
#--------------------------------------------------------# 
# Function used in  visulization1 
def PRISMA_rgb_image(ds_PRISMA, selected_date_PR, selected_date_str_PR):
    red_band = ds_PRISMA.sel(time=f'{selected_date_PR}T10:30:00.000000000', method='nearest')['band32']
    green_band = ds_PRISMA.sel(time=f'{selected_date_PR}T10:30:00.000000000', method='nearest')['band22']
    blue_band = ds_PRISMA.sel(time=f'{selected_date_PR}T10:30:00.000000000', method='nearest')['band11']
    rgb_array = xr.concat([red_band, green_band, blue_band], dim='band').transpose('y', 'x', 'band')
    rgb_array = rgb_array[::-1, :, :]

    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_array * 3)  # Adjust the multiplier as needed
    plt.title(f'PRISMA {selected_date_str_PR} RGB Image')
    plt.axis('off')
    plt.show()
        
def S2_rgb_image(ds_S2, selected_date_S, selected_date_str_S):
    extent = (488840.0,5014260.0,525740.0,5050440.0)
    red_band = ds_S2.sel(time=f'{selected_date_S}T10:30:00.000000000', method='nearest')['band3']
    red_band = red_band.sel(x=slice(extent[0], extent[2]), y=slice(extent[1], extent[3]))

    green_band = ds_S2.sel(time=f'{selected_date_S}T10:30:00.000000000', method='nearest')['band2']
    green_band = green_band.sel(x=slice(extent[0], extent[2]), y=slice(extent[1], extent[3]))

    blue_band = ds_S2.sel(time=f'{selected_date_S}T10:30:00.000000000', method='nearest')['band1']
    blue_band = blue_band.sel(x=slice(extent[0], extent[2]), y=slice(extent[1], extent[3]))

    rgb_array = xr.concat([red_band, green_band, blue_band], dim='band').transpose('y', 'x', 'band')
    rgb_array = rgb_array[::-1, :, :]

    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_array * 3)  # Adjust the multiplier as needed
    plt.title(f'S2 {selected_date_str_S} RGB Image')
    plt.axis('off')
    plt.show()
    
    
    
def visualize_training_samples(training, legend, cmm_gdf):
    training['LCZ'] = training['LCZ'].astype(int)
    training = training.sort_values('LCZ')
    
    # Add a column with the correspondence between LCZ class and its name
    training['LCZ_name'] = training['LCZ'].map(legend).str[0]
    
    lcz_list = [value[0] for value in legend.values()]
    cmap_colors = [value[1] for value in legend.values()]

    m = cmm_gdf.explore(
        style_kwds={'fillOpacity': 0},
        marker_kwds=dict(radius=10, fill=True),  
        tooltip_kwds=dict(labels=False),  
        tooltip=False,
        popup=False,
        highlight=False,
        name="cmm"  
    )

    plot=training.explore(
        m=m,
        column="LCZ_name",  
        tooltip="LCZ_name",  
        popup=True,  
        tiles="CartoDB positron",  
        style_kwds=dict(color="black"),  
        categories=lcz_list,
        cmap=cmap_colors
    )
    return plot
   
def compute_spectral_signature(combined_ds, training, legend):
    LCZ_classes = list(legend.keys())
    spectral_sign = {}
    spectral_sign_std = {}
    band_threshold = 1e-8

    shapes = {LCZ: gdf.geometry for LCZ, gdf in training.groupby('LCZ')}

    for LCZ in LCZ_classes:
        print(f'Computing spectral signature for LCZ class: {legend[LCZ][0]}')

        geometry = shapes[LCZ]

        clipped_ds = combined_ds.rio.clip(geometry, from_disk=True)

        class_data = clipped_ds.to_array().values
        class_data[class_data == 0] = np.nan

        spectral_sign[LCZ] = np.nanmedian(class_data, axis=(1, 2))
        spectral_sign_std[LCZ] = np.nanstd(class_data, axis=(1, 2))

        spectral_sign[LCZ] = spectral_sign[LCZ][spectral_sign[LCZ] > band_threshold]
        spectral_sign_std[LCZ] = spectral_sign_std[LCZ][spectral_sign_std[LCZ] > band_threshold]

        del clipped_ds

    return spectral_sign, spectral_sign_std

def plot_spectral_sign(sensor, wvl, selected_classes, spectral_sign, spectral_sign_std, legend):
    
    pl.figure(figsize=(14,6))
    for LCZ in sorted(selected_classes):
        pl.plot(sorted(wvl), spectral_sign[LCZ], label = legend[LCZ][0], color=legend[LCZ][1])
        pl.fill_between(sorted(wvl), spectral_sign[LCZ]-spectral_sign_std[LCZ], spectral_sign[LCZ]+spectral_sign_std[LCZ], alpha=.3, color=legend[LCZ][1])
    pl.xlabel('Wavelength (nm)')
    pl.ylabel('Reflectance')
    pl.title(f'Median spectral signature and confidence interval +/-sigma of the training samples computed from {sensor} image')
    pl.legend()
    
    
def update_plot(ds_classified_PRI,ds_classified_S2,sensor, prisma_date, s2_date):
    global img1
    
    
    if sensor == 'PRISMA':
        prisma_date = pd.to_datetime(prisma_date)
        classified = ds_classified_PRI.sel(time=f'{prisma_date}T10:30:00.000000000', method='nearest')['band']
        
    elif sensor == 'Sentinel-2':
        s2_date = pd.to_datetime(s2_date)
        classified = ds_classified_S2.sel(time=f'{s2_date}T10:30:00.000000000', method='nearest')['band']
        
    else:
        raise ValueError("Invalid sensor type. Supported sensors are 'PRISMA' and 'Sentinel-2'.")

    # Rest of your plotting code
    legend = {
                   2: ['Compact mid-rise', '#D10000'],
                   3: ['Compact low-rise', '#CD0000'],
                   5: ['Open mid-rise', '#FF6600'],
                   6: ['Open low-rise', '#FF9955'],
                   8: ['Large low-rise', '#BCBCBC'],
                   101: ['Dense trees', '#006A00'],
                   102: ['Scattered trees', '#00AA00'],
                   104: ['Low plants', '#B9DB79'],
                   105: ['Bare rock or paved', '#545454'],
                   106: ['Bare soil or sand', '#FBF7AF'],
                   107: ['Water', '#6A6AFF']
       } 

    cmap_colors = [legend[key][1] for key in legend.keys()]
    cmap1 = plt.cm.colors.ListedColormap(cmap_colors, name='LCZ classes colormap')
    bounds = [int(key) for key in legend.keys()]
    bounds.append(bounds[-1] + 1)
    norm1 = colors.BoundaryNorm(bounds, cmap1.N)

    fig2, ax1 = plt.subplots(figsize=(14, 14))
    plt.title('Classified LCZ image ')

    # Turn off the axes frame and labels
    ax1.axis('off')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Add a legend class colors
    labels = [legend[i][0] for i in legend.keys()]
    handle = [plt.Rectangle((0, 0), 1, 1, color=legend[key][1]) for key in list(legend.keys())]
    legend = plt.legend(handle, labels, title='LCZ Classes', bbox_to_anchor=(0.99, 1), loc='upper left')
    plt.setp(legend.get_title(), fontsize='12')  # Adjust the font size of the legend title
  
    img1 = ax1.imshow(classified, cmap=cmap1, interpolation='none', norm=norm1, origin='upper')
    plt.gca().invert_yaxis()  # to flip the-axis
    plt.show()
    buf = io.BytesIO()
    fig2.savefig(buf)
    buf.seek(0)
    img1 = Image.open(buf)
    return img1

def LCZ_PR_plot(PRISMA_Date, ds_classified_PRI):
    prisma_date = pd.to_datetime(PRISMA_Date)
    Classified = ds_classified_PRI.sel(time=f'{prisma_date}T10:30:00.000000000', method='nearest')['band']
    
    legend = {
        2: ['Compact mid-rise', '#D10000'],
        3: ['Compact low-rise', '#CD0000'],
        5: ['Open mid-rise', '#FF6600'],
        6: ['Open low-rise', '#FF9955'],
        8: ['Large low-rise', '#BCBCBC'],
        101: ['Dense trees', '#006A00'],
        102: ['Scattered trees', '#00AA00'],
        104: ['Low plants', '#B9DB79'],
        105: ['Bare rock or paved', '#545454'],
        106: ['Bare soil or sand', '#FBF7AF'],
        107: ['Water', '#6A6AFF']
    } 
            
    cmap_colors = [legend[key][1] for key in legend.keys()]
    cmap1 = plt.cm.colors.ListedColormap(cmap_colors, name='LCZ classes colormap')
    bounds = [int(key) for key in legend.keys()]
    bounds.append(bounds[-1] + 1)
    norm1 = colors.BoundaryNorm(bounds, cmap1.N)
    
    fig2 = plt.figure(figsize=(14, 14))
    ax1 = fig2.add_subplot(1, 1, 1)
    plt.title(f'{PRISMA_Date}', x=0, y=1.02)
    
    # Turn off the axes frame and labels
    ax1.axis('off')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add a legend class colors
    labels = [legend[i][0] for i in legend.keys()]
    handle = [plt.Rectangle((0, 0), 1, 1, color=legend[key][1]) for key in list(legend.keys())]
    legend = plt.legend(handle, labels, title='LCZ Classes', bbox_to_anchor=(-0.11, 1), loc='upper left')
    plt.setp(legend.get_title(), fontsize='12')  

    img1 = ax1.imshow(Classified, cmap=cmap1, interpolation='none', norm=norm1, origin='upper')
    plt.gca().invert_yaxis()  # to flip the y-axis
    

    buf = io.BytesIO()
    fig2.savefig(buf)
    buf.seek(0)
    

    img1 = Image.open(buf)
    
    plt.close(fig2)  
    
    return img1


def LCZ_S2_plot(S2_Date, ds_classified_S2):
    s2_date = pd.to_datetime(S2_Date)
    Classified = ds_classified_S2.sel(time=f'{s2_date }T10:30:00.000000000', method='nearest')['band']
    
    legend = {
        2: ['Compact mid-rise', '#D10000'],
        3: ['Compact low-rise', '#CD0000'],
        5: ['Open mid-rise', '#FF6600'],
        6: ['Open low-rise', '#FF9955'],
        8: ['Large low-rise', '#BCBCBC'],
        101: ['Dense trees', '#006A00'],
        102: ['Scattered trees', '#00AA00'],
        104: ['Low plants', '#B9DB79'],
        105: ['Bare rock or paved', '#545454'],
        106: ['Bare soil or sand', '#FBF7AF'],
        107: ['Water', '#6A6AFF']
    } 
            
    cmap_colors = [legend[key][1] for key in legend.keys()]
    cmap1 = plt.cm.colors.ListedColormap(cmap_colors, name='LCZ classes colormap')
    bounds = [int(key) for key in legend.keys()]
    bounds.append(bounds[-1] + 1)
    norm1 = colors.BoundaryNorm(bounds, cmap1.N)
    
    fig2 = plt.figure(figsize=(14, 14))
    ax1 = fig2.add_subplot(1, 1, 1)
    plt.title(f'{S2_Date}', x=1, y=1.02)
    
    # Turn off the axes frame and labels
    ax1.axis('off')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add a legend class colors
    labels = [legend[i][0] for i in legend.keys()]
    handle = [plt.Rectangle((0, 0), 1, 1, color=legend[key][1]) for key in list(legend.keys())]
    legend = plt.legend(handle, labels, title='LCZ Classes', bbox_to_anchor=(-0.1, 1), loc='upper left')
    plt.setp(legend.get_title(), fontsize='12')  

    img2 = ax1.imshow(Classified, cmap=cmap1, interpolation='none', norm=norm1, origin='upper')
    plt.gca().invert_yaxis()  
 
    buf = io.BytesIO()
    fig2.savefig(buf)
    buf.seek(0)
    
 
    img2 = Image.open(buf)
    
    plt.close(fig2) 
    
    return img2

def update_climami_plot(ds_climami,extent,raster_name):
    global img2
    #raster_name= raster_name.value
    climami = ds_climami.sel(time='2023-02-09T10:30:00.000000000', method='nearest')[raster_name.replace('.tif', '')]

    # Clip the raster based on the specified extent
    climami_clipped = climami.sel(
       x=slice(extent[0], extent[2]),
       y=slice(extent[1], extent[3])
    )
    # Get the minimum and maximum values for normalization
    vmin = climami_clipped.min().item()
    vmax = climami_clipped.max().item()

    # Create a colormap, normalization, and ScalarMappable
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(14, 14))

    # Plot the raster using imshow with origin='upper'
    im = ax.imshow(climami_clipped, cmap=cmap, norm=norm, origin='upper')

    # Create a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="1%", pad=0.8)
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Temperature')

    # Turn off the axes frame and labels
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    # Invert the y-axis
    ax.invert_yaxis()
    plt.title('ClimaMI Map')
    plt.show()

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img2 = Image.open(buf)
    
def selected_lcz(ds_classified_PRI,ds_classified_S2,extent,sensor, prisma_date, s2_date):
    
    
    
    if sensor == 'PRISMA':
        prisma_date = pd.to_datetime(prisma_date)
        classified = ds_classified_PRI.sel(time=f'{prisma_date}T10:30:00.000000000', method='nearest')['band']
        
    elif sensor == 'Sentinel-2':
        s2_date = pd.to_datetime(s2_date)
        classified = ds_classified_S2.sel(time=f'{s2_date}T10:30:00.000000000', method='nearest')['band']
        
    else:
        raise ValueError("Invalid sensor type. Supported sensors are 'PRISMA' and 'Sentinel-2'.")

    # Rest of your plotting code
    legend = {
                   2: ['Compact mid-rise', '#D10000'],
                   3: ['Compact low-rise', '#CD0000'],
                   5: ['Open mid-rise', '#FF6600'],
                   6: ['Open low-rise', '#FF9955'],
                   8: ['Large low-rise', '#BCBCBC'],
                   101: ['Dense trees', '#006A00'],
                   102: ['Scattered trees', '#00AA00'],
                   104: ['Low plants', '#B9DB79'],
                   105: ['Bare rock or paved', '#545454'],
                   106: ['Bare soil or sand', '#FBF7AF'],
                   107: ['Water', '#6A6AFF']
       } 

    cmap_colors = [legend[key][1] for key in legend.keys()]
    cmap1 = plt.cm.colors.ListedColormap(cmap_colors, name='LCZ classes colormap')
    bounds = [int(key) for key in legend.keys()]
    bounds.append(bounds[-1] + 1)
    norm1 = colors.BoundaryNorm(bounds, cmap1.N)

    fig2, ax1 = plt.subplots(figsize=(14, 14))
    plt.title('Classified LCZ image ')

    # Turn off the axes frame and labels
    ax1.axis('off')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Add a legend class colors
    labels = [legend[i][0] for i in legend.keys()]
    handle = [plt.Rectangle((0, 0), 1, 1, color=legend[key][1]) for key in list(legend.keys())]
    legend = plt.legend(handle, labels, title='LCZ Classes', bbox_to_anchor=(0.99, 1), loc='upper left')
    plt.setp(legend.get_title(), fontsize='12')  # Adjust the font size of the legend title
  
    img3 = ax1.imshow(classified, cmap=cmap1, interpolation='none', norm=norm1, origin='upper')
    plt.gca().invert_yaxis()  # to flip the-axis

    buf = io.BytesIO()
    fig2.savefig(buf)
    buf.seek(0)
    img3 = Image.open(buf)
    plt.close(fig2)
    return img3
    
    
    
def selected_climami(ds_climami,extent,raster_name):
    
    #raster_name= raster_name.value
    climami = ds_climami.sel(time='2023-02-09T10:30:00.000000000', method='nearest')[raster_name.replace('.tif', '')]

    # Clip the raster based on the specified extent
    climami_clipped = climami.sel(
       x=slice(extent[0], extent[2]),
       y=slice(extent[1], extent[3])
    )
    # Get the minimum and maximum values for normalization
    vmin = climami_clipped.min().item()
    vmax = climami_clipped.max().item()

    # Create a colormap, normalization, and ScalarMappable
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(14, 14))

    # Plot the raster using imshow with origin='upper'
    img4 = ax.imshow(climami_clipped, cmap=cmap, norm=norm, origin='upper')

    # Create a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="1%", pad=0.8)
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Temperature')

    # Turn off the axes frame and labels
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    # Invert the y-axis
    ax.invert_yaxis()
    plt.title('ClimaMI Map')
   

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img4 = Image.open(buf)  
    plt.close(fig)
    return img4


def process_zonal_stats(Classified, Climami_clipped, legend):
    """
    Process zonal statistics and return a DataFrame with cleaned data.

    Parameters:
    - Classified (DataFrame): DataFrame representing classified data.
    - Climami_clipped (DataFrame): DataFrame representing climatic data.
    - legend (dict): Dictionary containing information for the legend.

    Returns:
    - result_t (DataFrame): Processed DataFrame with zonal statistics.
    """
    result = zonal_stats(Classified, Climami_clipped)
    result_t = result.drop(['sum', 'var'], axis=1)
    result_t = result_t[['zone', 'count', 'min', 'max', 'mean', 'std']]
    result_t['count'] = result_t['count'].astype(int)
    result_t['zone'] = result_t['zone'].replace(legend.keys(), [legend[key][0] for key in legend.keys()])
    result_t = result_t.round(2)
    
    return result_t



def create_boxplot_data(Climami_clipped, Classified):
    # Flatten the arrays
    climami_array = Climami_clipped.values.flatten()
    classified_array = Classified.values.flatten()

    # Create DataFrame
    df_box = pd.DataFrame({
        'zone': classified_array,
        'Temperature': climami_array
    })

    # Filter out rows where the zone is 'nan'
    df_box = df_box[df_box['zone'] != 'nan']

    return df_box




def create_boxplot(df_box, legend):
    """
    Create a boxplot based on the provided DataFrame and legend.
    
    Parameters:
    - df_box (DataFrame): DataFrame containing 'zone' and 'Temperature' columns.
    - legend (dict): Dictionary containing information for the legend.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_box, x='zone', y='Temperature', palette=[legend[zone][1] for zone in sorted(legend.keys())])
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Temperature', fontsize=14)
    plt.title('Boxplot for each LCZ Class', fontsize=16)
    plt.xticks(ticks=range(len(legend)), labels=[legend[zone][0] for zone in sorted(legend.keys())], rotation=45, ha='right')
    plt.legend(title='Statistic', fontsize=12, title_fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
