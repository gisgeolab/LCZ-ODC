# Authors: Alberto Vavassori, Emanuele Capizzi - DICA - GISGeolab - Politecnico di Milano, 2023.

#------------------------------------------------#

# Function used in 1 - S2_Preprocessing.ipynb

import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_TMR_TNR_paths(images_folder_path, sel_resol, sel_date):
    """
    Get the file paths for the TMR and TNR tiles in a Sentinel-2 imagery folder.

    Parameters
    ----------
    images_folder_path : str
        The path to the folder containing the unzipped folders of Sentinel-2 T32TMR and T32TNR tiles.
    sel_resol : str
        The desired resolution folder name, such as "R10m" or "R20m".

    Returns
    -------
    tuple
        A tuple containing the file paths to the TMR and TNR tiles.
    """
    sel_date = sel_date.replace('-', '')
    s2_folders = []
    tiles_tags = ['T32TMR', 'T32TNR']

    # Find all folders in the specified path that contain T32TMR or T32TNR in their names
    for filename in os.listdir(images_folder_path):
        if any(tile in filename for tile in tiles_tags) and (sel_date in filename):
            s2_folders.append(filename)

    # Find the names of the T32TMR and T32TNR folders
    TMR_folder_name = [elem for elem in s2_folders if tiles_tags[0] in elem][0]
    TNR_folder_name = [elem for elem in s2_folders if tiles_tags[1] in elem][0]

    # Construct the file paths for the desired resolution folders in the T32TMR and T32TNR folders
    path_TMR = os.path.join(images_folder_path, TMR_folder_name, TMR_folder_name+".SAFE", "GRANULE", os.listdir(os.path.join(images_folder_path, TMR_folder_name, TMR_folder_name+".SAFE", "GRANULE"))[0], 'IMG_DATA', sel_resol)
    path_TNR = os.path.join(images_folder_path, TNR_folder_name, TNR_folder_name+".SAFE", "GRANULE", os.listdir(os.path.join(images_folder_path, TNR_folder_name, TNR_folder_name+".SAFE", "GRANULE"))[0], 'IMG_DATA', sel_resol)

    print(f"TMR path to {sel_resol} folder: {path_TMR}")
    print(f"TNR path to {sel_resol} folder: {path_TNR}")

    return (path_TMR, path_TNR)

#------------------------------------------------#

def get_PRISMA_path(prisma_images_folder_path, sel_prisma_date):
    """
    Get the path to the PRISMA image file for a specific date.

    Args:
        prisma_images_folder_path (str): Path to the folder containing PRISMA image files.
        sel_prisma_date (str): Selected PRISMA date in the format 'YYYY-MM-DD'.

    Returns:
        str: Path to the PRISMA image file for the specified date, or None if not found.
    """
    
    sel_date = sel_prisma_date.replace('-', '')
    tiles_tags = ['PRS_L2D_STD'] 

    # Find all folders in the specified path that contain T32TMR or T32TNR in their names
    for filename in os.listdir(prisma_images_folder_path):
        if any(tile in filename for tile in tiles_tags) and (sel_date in filename):
            return os.path.join(prisma_images_folder_path, filename)
        
#------------------------------------------------#

def get_s2_path(prisma_images_folder_path, sel_prisma_date):
    """
    Get the path to the Sentinel-2 image file for a specific date.

    Args:
        prisma_images_folder_path (str): Path to the folder containing Sentinel-2 image files.
        sel_prisma_date (str): Selected date in the format 'YYYY-MM-DD'.

    Returns:
        str: Path to the Sentinel-2 image file for the specified date, or None if not found.
    """

    sel_date = sel_prisma_date.replace('-', '')
    tiles_tags = ['S2']
    expected_format = 'S2_{}_20m_clip.tif'.format(sel_date)

    # Find the file with the expected format in the specified path
    for filename in os.listdir(prisma_images_folder_path):
        if filename == expected_format:
            return os.path.join(prisma_images_folder_path, filename)
    
    return None
        
#------------------------------------------------#

# Function used in 1 - S2_Preprocessing.ipynb

import rasterio
import numpy as np
import copy

def convert_jp2_to_geotiff(folder_path, band_names, output_path):
    """
    Converts a set of Sentinel-2 JP2 files to a single GeoTIFF file with multiple bands.
    
    Parameters
    ----------
    folder_path : str
        The path to the folder containing the JP2 files.
    band_names : list of str
        The names of the bands to include in the output file.
    output_path : str
        The path to the output GeoTIFF file.
    
    Returns
    -------
    None
    
    """
    # Get a list of all files in the folder that contain any of the band names
    list_files = []
    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        if any(band in filename for band in band_names):
            list_files.append(f)
    
    # Get the number of bands to include in the output file
    count = len(list_files)
    
    # Create an empty list to store the band data
    band_data = []
    
    # Loop over the list of files and read each band into memory
    for file in list_files:
        # Open the JP2 file
        with rasterio.open(file) as jp2_file:
            # Append the band data to the list
            band_data.append(jp2_file.read())
            
            # If this is the first band, update the profile to match the file
            if len(band_data) == 1:
                profile = jp2_file.profile
                profile['driver'] = 'GTiff'
                profile['count'] = count
                profile['dtype'] = 'float32'
    
    # Stack the band data into a single array and remove the first axis (band axis)
    stacked = np.stack(band_data).squeeze().astype('float32')
    
    # Write the stacked band data to the output file
    with rasterio.open(output_path, 'w', **profile) as output_file:
        output_file.write(stacked)
        print(f"GeoTIFF named {output_path} with bands {band_names} has been created!")

#------------------------------------------------#
# Function used in 1 - S2_Preprocessing.ipynb

from rasterio.merge import merge

def merge_tiles_s2(tiles, output_path, epsg="32632"):
    
    """
    Merge multiple Sentinel-2 tiles into a single mosaic image and write the output to a GeoTIFF file. Default EPSG is 32632.
    
    Parameters
    ----------
    tiles (list): A list of file paths representing the tiles to merge.
    output_path (str): The output file path to write the merged mosaic image.

    Returns
    -------
    """
    
    src_files_to_mosaic = []
    
    for tile in tiles:
        src = rasterio.open(tile)
        src_files_to_mosaic.append(src)
    
    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Copy the metadata
    out_meta = src.meta.copy()
    
    # Update metadata with merged array properties and projection info
    out_meta.update({"driver": "GTiff",
     "height": mosaic.shape[1],
     "width": mosaic.shape[2],
     "transform": out_trans,
     "crs": "epsg:"+epsg
     }
    )
    
    # Write the merged mosaic array to a GeoTIFF file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write((mosaic - 1000) / 10000)

#------------------------------------------------#
# Function used in 2 - PRISMA_S2_Coregistration.ipynb
# and in 5 - Classification.ipynb

import geopandas as gpd
from rasterio.mask import mask

def clip_image_study_area(input_geotiff, output_geotiff, study_area):
    # Open the GeoTIFF file in read mode
    with rasterio.open(input_geotiff) as src:
        # Clip the GeoTIFF using the vector file as a mask
        out_image, out_transform = mask(src, study_area.geometry, crop=True, all_touched=True)
        out_meta = src.meta

    # Update the metadata for the output GeoTIFF
    out_meta.update({
        'driver': 'GTiff',
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform
    })

    # Write the clipped image to the output GeoTIFF file
    with rasterio.open(output_geotiff, 'w', **out_meta) as dst:
        dst.write(out_image)
        print(output_geotiff + " created!")
        

#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

def get_prisma_wvl(prisma_meta):
    
    # Retrieve PRISMA central wavelengths
    vnir_bands = prisma_meta.attrs['List_Cw_Vnir']
    swir_bands = prisma_meta.attrs['List_Cw_Swir']
    vnir_dict = {}
    swir_dict = {}
    wvl_dict = {}
    
    for i, band in enumerate(vnir_bands):
        vnir_dict[len(vnir_bands) - i] = band
    for i, band in enumerate(swir_bands):
        swir_dict[len(swir_bands)+len(vnir_bands) - i] = band
    
    
    vnir_dict_sorted = dict(sorted(vnir_dict.items(), key=lambda x: x[1], reverse=False)) #reverse dictionary from lower to higher keys
    swir_dict_sorted = dict(sorted(swir_dict.items(), key=lambda x: x[1], reverse=False)) #reverse dictionary from lower to higher keys
    
    wvl_dict = {**vnir_dict, **swir_dict} #group together the two dictionaries
    wvl_dict_sorted = dict(sorted(wvl_dict.items(), key=lambda x: x[1], reverse=False)) #reverse dictionary from lower to higher keys
    
    wvl_dict_sorted = {key: value for key, value in wvl_dict_sorted.items() if value != 0} #remove 0 wvls
    wvl_dict_sorted = {i: value for i, (_, value) in enumerate(wvl_dict_sorted.items())} #re-number keys from 0 to 233
    
    
    vnir_dict = {x:y for x,y in vnir_dict.items() if y!=0}
    swir_dict = {x:y for x,y in swir_dict.items() if y!=0}
    
    vnir_wvl_values = list(vnir_dict.values())
    swir_wvl_values = list(swir_dict.values())
    
    wvl_decr = swir_wvl_values + vnir_wvl_values
    wvl = wvl_decr[::-1] #reverse order of bands

    
    return wvl, wvl_dict_sorted


#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

import matplotlib.gridspec as gridspec
import ipywidgets as widgets
import matplotlib.pylab as pl

def plot_signature_widgets(selected_prisma_image, wvl, wvl_s, data, data_s):
    
    # Prisma image
    with rasterio.open(selected_prisma_image) as src:
        red = src.read(32)
        green = src.read(22)
        blue = src.read(11)
        bands = src.read()

    # Plot the RGB image and fix y direction
    rgb = np.stack((red, green, blue), axis=2)*6
    rgb = np.flipud(rgb)
    # Scale pixel values to [0, 1] range
    rgb = np.clip(rgb, 0, 1)

    # Setup grid
    gs = gridspec.GridSpec(2, 2)

    def plot(x, y):
        pl.figure(figsize=(14,8))
    
        # Plot the rgb image in the upper left corner
        ax = pl.subplot(gs[0, 0]) # row 0, col 0
        pl.imshow(rgb)
        pl.plot(x, y, marker="+", markersize=15, markerfacecolor="red", markeredgecolor="red", mew=2)
        ax.invert_yaxis()

        ax = pl.subplot(gs[0, 1]) # row 0, col 1
        # Plot the zoomed image in the upper right corner
        pl.imshow(rgb[y-50:y+50, x-50:x+50])
        pl.plot(50, 50, marker="+", markersize=15, markerfacecolor="red", markeredgecolor="red", mew=2)
        ax.invert_yaxis()
    
        # Plot the spectral signature below
        ax = pl.subplot(gs[1, :]) # row 1, span all columns
        pl.plot(sorted(wvl), data[:, x, y], label = 'PRISMA')
        pl.plot(wvl_s, data_s[:, x, y], label = 'Sentinel-2')
        #plt.xticks(range(len(wvl)), [round(w, 2) for w in wvl], rotation=45)
        #plt.gca().xaxis.set_major_locator(MultipleLocator(8))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.legend()

    # Sliders
    x_slider = widgets.IntSlider(value=600, min=0, max=data.shape[1]-1, step=1)
    y_slider = widgets.IntSlider(value=600, min=0, max=data.shape[2]-1, step=1, orientation='vertical')

    # Interactivy
    widgets.interact(plot, x = x_slider, y = y_slider)

    plt.show()

    
#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

def plot_training_samples(training_folder, cmm_folder, legend):
    """
    Function to plot the spatial distribution of the training samples in the area of interest (i.e. Metropolitan City of Milan).
    Args:
        training_folder (str): path to the geopackage with the boundaries of the training samples
        cmm_folder (str): path to the geopackage with the boundaries of the Metropolitan City of Milan
        colors_dict (dict): dictionary containing the colors per LCZ
        legend (dict): dictionary containing the class name per LCZ

    Returns:
        training (dataframe): geodataframe with the geometries of the training samples
        m (folium map): folium map with the plot of training samples
        shapes (dict): dictionary containing the geometries of the training samples
        LCZ_class (array): array with the LCZ classes

    """
    
    cmm_gdf = gpd.read_file(cmm_folder)
    training = gpd.read_file(training_folder)
    
    training['LCZ'] = training['LCZ'].astype(int)
    training = training.sort_values('LCZ')
    
    # add a column with the correspondence between LCZ class and its name
    training['LCZ_name'] = training['LCZ'].map(legend).str[0]
    
    lcz_list = [value[0] for value in legend.values()]
    
    cmap_colors = [value[1] for value in legend.values()]
    
    print(f'List of LCZ: {lcz_list}')
    print(f'List of colors: {cmap_colors}')

    m = cmm_gdf.explore(
        style_kwds = {'fillOpacity': 0},
        marker_kwds=dict(radius=10, fill=True), # make marker radius 10px with fill
        tooltip_kwds=dict(labels=False), # do not show column label in the tooltip
        tooltip = False, 
        popup = False,
        highlight = False,
        name="cmm" # name of the layer in the map
    )

    training.explore(m=m, 
                     column="LCZ_name", # make choropleth based on "BoroName" column
                     tooltip="LCZ_name", # show "BoroName" value in tooltip (on hover)
                     popup=True, # show all values in popup (on click)
                     tiles="CartoDB positron", # use "CartoDB positron" tiles
                     style_kwds=dict(color="black"), # use black outline
                     categories=lcz_list,
                     cmap=cmap_colors
                    )
    
    # create a dictionary (shapes) containing the geometries of the training samples
    # the dictionary keys are the LCZ classes
    shapes = {}
    LCZ_class = training['LCZ'].unique()
    for LCZ in LCZ_class:
        shapes[LCZ] = training.loc[training['LCZ'] == LCZ].geometry
    
    return training, m, shapes


#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

def compute_spectral_signature(image, legend, shapes):
    """
    Function to compute the median spectral signature of the training samples starting from the values of the satellite image.
    Args:
        image (numpy array): satellite image from which the signatures are computed
        LCZ_class (str): list containing the LCZ classes in the training samples
        shapes (dict): dictionary containing the geometries of the training samples

    Returns:
        spectral_sign (dict): dictionary with the median of reflectance, the keys are the LCZ classes
        spectral_sign_std (dict): dictionary with the standard deviation of reflectance, the keys are the LCZ classes

    """
    
    LCZ_class = list(legend.keys())
    
    # clip the PRISMA image to the polygon extent and compute the spectral signature
    band_threshold = 1e-8
    spectral_sign_median = {}
    spectral_sign_mean = {}
    spectral_sign_std = {}
    pixels_spectral_sign_raw = {}
    pixels_spectral_sign = {}
    with rasterio.open(image) as src:
        for LCZ in LCZ_class:
            print(f'Computed spectral signature statistics in the training samples for class: {legend[LCZ][0]}')
            out_image, out_transform = rasterio.mask.mask(dataset=src, shapes=shapes[LCZ], crop=True, pad=True)
            out_image[out_image == 0] = np.nan
        
            # compute median, mean, std
            spectral_sign_median[LCZ] = np.nanmedian(out_image, axis=(1, 2))
            spectral_sign_mean[LCZ] = np.nanmean(out_image, axis=(1, 2))
            spectral_sign_std[LCZ] = np.nanstd(out_image, axis=(1, 2))
            
    return spectral_sign_median, spectral_sign_mean, spectral_sign_std#, pixels_spectral_sign


#------------------------------------------------#
def compute_pixel_spectral_signature(image, legend, shapes):
    
    LCZ_class = list(legend.keys())
    
    # clip the PRISMA image to the polygon extent and compute the spectral signature
    band_threshold = 1e-8
    pixels_spectral_sign_raw = {}
    pixels_spectral_sign = {}
    with rasterio.open(image) as src:
        for LCZ in LCZ_class:
            print(f'Computed pixel-wise spectral signature in the training samples for class: {legend[LCZ][0]}')
            out_image, out_transform = rasterio.mask.mask(dataset=src, shapes=shapes[LCZ], crop=True, pad=True)
            out_image[out_image == 0] = np.nan
            
            # extract single pixel signatures
            flattened_image = out_image.reshape(out_image.shape[0], -1)
            flattened_image = flattened_image.T
            
            # put the single pixel signatures in the dictionary
            if LCZ not in pixels_spectral_sign_raw:
                pixels_spectral_sign_raw[LCZ] = flattened_image
            else:
                pixels_spectral_sign_raw[LCZ] = np.concatenate((pixels_spectral_sign_raw[LCZ], flattened_image), axis=0)
            
            # remove all the pixels with nan values (the ones belonging to the bounding box of each specific class training sample, yet not containing training samples)
            non_nan_rows_mask = ~np.all(np.isnan(pixels_spectral_sign_raw[LCZ]), axis=1)
            pixels_spectral_sign[LCZ] = pixels_spectral_sign_raw[LCZ][non_nan_rows_mask]
            
            # set to zero all the band values having nan, because it causes problems with spectral signatures statistics (especially for water)
            nan_columns = np.isnan(pixels_spectral_sign[LCZ]).any(axis=0)
            pixels_spectral_sign[LCZ][:, nan_columns] = np.nan_to_num(pixels_spectral_sign[LCZ][:, nan_columns], nan = 0)
            
    return pixels_spectral_sign


#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

import plotly.graph_objects as go

def plot_spectral_sign(sensor, wvl, selected_classes, spectral_sign, spectral_sign_std, legend):
    
    scale_w = widgets.RadioButtons(
        options = ['Do NOT scale to 0-1', 'Scale to 0-1'],
        description = 'How to display the y axis?',
        style = {'description_width': 'initial'},
        disabled = False,
        continuous_update = False
    )
    
    def plot_spectral_sign_widgets(scale):
        if sensor == 'PRISMA':
            fig = go.Figure()
            for LCZ in sorted(selected_classes):
                fig.add_trace(go.Scatter(x=sorted(wvl), y=spectral_sign[LCZ], mode='lines', name=legend[LCZ][0], line=dict(color=legend[LCZ][1])))
                fig.add_trace(go.Scatter(x=sorted(wvl), y=spectral_sign[LCZ] - spectral_sign_std[LCZ], mode='lines', name=f'Lower Bound {legend[LCZ][0]}', line=dict(color=legend[LCZ][1], dash='dash')))
                fig.add_trace(go.Scatter(x=sorted(wvl), y=spectral_sign[LCZ] + spectral_sign_std[LCZ], mode='lines', name=f'Upper Bound {legend[LCZ][0]}', line=dict(color=legend[LCZ][1], dash='dash')))

            fig.update_xaxes(title='Wavelength (nm)')
            fig.update_yaxes(title='Reflectance')
            if scale_w.value == 'Scale to 0-1':
                fig.update_layout(title=f'Median spectral signature and confidence interval +/-sigma of the training samples computed from {sensor} image', yaxis_range=[0, 1], showlegend=True, width=1000, height=500)
            else:
                fig.update_layout(title=f'Median spectral signature and confidence interval +/-sigma of the training samples computed from {sensor} image', showlegend=True, width=1000, height=500)
            fig.show()

        elif sensor == 'PRISMA-pansharpened':
            fig = go.Figure()
            for LCZ in sorted(selected_classes):
                fig.add_trace(go.Scatter(x=sorted(wvl), y=spectral_sign[LCZ], mode='lines', name=legend[LCZ][0], line=dict(color=legend[LCZ][1])))
                fig.add_trace(go.Scatter(x=sorted(wvl), y=spectral_sign[LCZ] - spectral_sign_std[LCZ], mode='lines', name=f'Lower Bound {legend[LCZ][0]}', line=dict(color=legend[LCZ][1], dash='dash')))
                fig.add_trace(go.Scatter(x=sorted(wvl), y=spectral_sign[LCZ] + spectral_sign_std[LCZ], mode='lines', name=f'Upper Bound {legend[LCZ][0]}', line=dict(color=legend[LCZ][1], dash='dash')))

            fig.update_xaxes(title='Wavelength (nm)')
            fig.update_yaxes(title='Reflectance')
            if scale_w.value == 'Scale to 0-1':
                fig.update_layout(title=f'Median spectral signature and confidence interval +/-sigma of the training samples computed from {sensor} image', yaxis_range=[0, 1], showlegend=True, width=1000, height=500)
            else:
                fig.update_layout(title=f'Median spectral signature and confidence interval +/-sigma of the training samples computed from {sensor} image', showlegend=True, width=1000, height=500)
            fig.show()
    
    interactive_plot = widgets.interact(plot_spectral_sign_widgets, scale = scale_w)


#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

def plot_spectral_sign_mean_median(sensor, wvl, selected_classes, spectral_sign_mean, spectral_sign_median, legend):
    
    scale_w = widgets.RadioButtons(
        options = ['Do NOT scale to 0-1', 'Scale to 0-1'],
        description = 'How to display the y axis?',
        style = {'description_width': 'initial'},
        disabled = False,
        continuous_update = False
    )
    
    def plot_spectral_sign_mean_median_widgets(scale):
        if sensor == 'PRISMA':
            fig = go.Figure()
            for LCZ in sorted(selected_classes):
                fig.add_trace(go.Scatter(x=sorted(wvl), y=spectral_sign_mean[LCZ], mode='lines', name=legend[LCZ][0] + ' - mean', line=dict(color=legend[LCZ][1])))
                fig.add_trace(go.Scatter(x=sorted(wvl), y=spectral_sign_median[LCZ], mode='lines', name=legend[LCZ][0] + ' - median', line=dict(color=legend[LCZ][1], dash='dash')))
            fig.update_xaxes(title='Wavelength (nm)')
            fig.update_yaxes(title='Reflectance')
            if scale_w.value == 'Scale to 0-1':
                fig.update_layout(title=f'Mean and median spectral signature of the training samples computed from {sensor} image', showlegend=True, yaxis_range=[0, 1], width=1000, height=500)
            else:
                fig.update_layout(title=f'Mean and median spectral signature of the training samples computed from {sensor} image', showlegend=True, width=1000, height=500)
        
        elif sensor == 'PRISMA-pansharpened':
            fig = go.Figure()
            for LCZ in sorted(selected_classes):
                fig.add_trace(go.Scatter(x=sorted(wvl), y=spectral_sign_mean[LCZ], mode='lines', name=legend[LCZ][0] + ' - mean', line=dict(color=legend[LCZ][1])))
                fig.add_trace(go.Scatter(x=sorted(wvl), y=spectral_sign_median[LCZ], mode='lines', name=legend[LCZ][0] + ' - median', line=dict(color=legend[LCZ][1], dash='dash')))
            fig.update_xaxes(title='Wavelength (nm)')
            fig.update_yaxes(title='Reflectance')
            if scale_w.value == 'Scale to 0-1':
                fig.update_layout(title=f'Mean and median spectral signature of the training samples computed from {sensor} image', showlegend=True, yaxis_range=[0, 1], width=1000, height=500)
            else:
                fig.update_layout(title=f'Mean and median spectral signature of the training samples computed from {sensor} image', showlegend=True, width=1000, height=500)

        fig.show()
    
    interactive_plot = widgets.interact(plot_spectral_sign_mean_median_widgets, scale = scale_w)


#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

from plotly import graph_objs as go

def plot_spectral_sign_comparison(wvl, spectral_sign, spectral_sign_s, legend, selected_classes):


    scale_w = widgets.RadioButtons(
        options = ['Do NOT scale to 0-1', 'Scale to 0-1'],
        description = 'How to display the y axis?',
        style = {'description_width': 'initial'},
        disabled = False,
        continuous_update = False
    )
    
    def plot_spectral_sign_comparison_widgets(scale):
        fig = go.Figure()
        for LCZ in sorted(selected_classes):
            fig.add_trace(go.Scatter(x = sorted(wvl),
                                    y=spectral_sign_s[LCZ],
                                    mode = 'lines',
                                    line = dict(color = legend[LCZ][1], width = 2, dash='dash'),
                                    name = f"PRISMA pansharpened - {legend[LCZ][0]}"))
            # fig.add_trace(go.Scatter(x = sorted(wvl),
            #                         y=spectral_sign_s[LCZ],
            #                         mode = 'markers',
            #                         marker = dict(symbol = 'circle', size = 8, color = legend[LCZ][1]),
            #                         name = f"PRISMA pansharpened - {legend[LCZ][0]}"))
            fig.add_trace(go.Scatter(x = sorted(wvl),
                                    y=spectral_sign[LCZ],
                                    mode = 'lines',
                                    line = dict(color = legend[LCZ][1], width = 2),
                                    name = f"PRISMA - {legend[LCZ][0]}"))
            # fig.add_trace(go.Scatter(x = sorted(wvl),
            #                         y=spectral_sign[LCZ],
            #                         mode = 'markers',
            #                         marker = dict(symbol = 'cross', size = 8, color = legend[LCZ][1]),
            #                         name = f"PRISMA pansharpened - {legend[LCZ][0]}"))
        fig.update_xaxes(title_text = "Wavelength (nm)")
        fig.update_yaxes(title_text = "Reflectance")
        if scale_w.value == 'Scale to 0-1':
            fig.update_layout(width = 1000, height = 600, yaxis_range=[0, 1], title = 'Median spectral signature of the training samples')
        else:
            fig.update_layout(width = 1000, height = 600, title = 'Median spectral signature of the training samples - comparison PRISMA/PRISMA-pan')

        fig.show()

    interactive_plot = widgets.interact(plot_spectral_sign_comparison_widgets, scale = scale_w)
    
    
#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

from matplotlib.ticker import MultipleLocator

def boxplot_training_samples(image, shapes, legend, wvl_dict):
    
    lcz_class_w = widgets.RadioButtons(
        options=[value[0] for value in legend.values()],
        description='Select the LCZ class:',
        style = {'description_width': 'initial'},
        disabled=False)
    
    def boxplot_training_samples_widgets(lcz):
        
        lcz_class = next(key for key, values in legend.items() if values[0] == lcz_class_w.value)
        band_threshold = 1e-8
        with rasterio.open(image) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes[lcz_class], crop=True, pad=True)
#             #out_image[out_image < band_threshold] = np.nan
            out_image = out_image[~np.all(out_image <= band_threshold, axis=(1,2))]
            


        # store the reflectance values in a Pandas dataframe (each column is a band)
        n_bands = out_image.shape[0]
        flat_image_data = np.reshape(out_image, (n_bands, -1)).T

        df = pd.DataFrame(flat_image_data)
        df = df.rename(columns = wvl_dict)
        df = df.rename(columns={col: f'{round(col, 1):.1f}' for col in df.columns})

        new_df = df.stack().reset_index()
        new_df.drop(['level_0'], axis=1, inplace = True)
        new_df.rename(columns={"level_1": "band", 0: "values"}, inplace = True)

        #new_df.loc[~(new_df['values']==0).all(axis=1)]
        new_df = new_df.loc[~(new_df['values'] == 0)]

        fig, ax1 = pl.subplots(1, sharex = True, figsize=(14,6))

        PROPS = {
            'boxprops':{'edgecolor':'black'},
            'medianprops':{'color':'black'},
            'flierprops':{'marker': '.', 'markerfacecolor': 'grey', 'markersize': 2}
        }

        #new_df.columns = new_df.columns.astype(float)
        new_df['band'] = new_df['band'].astype(float)
        sns.boxplot(x = 'band', y = 'values', data=new_df, color = legend[lcz_class][1], dodge = True, width = 0.8, ax = ax1, **PROPS)

        #plt.xticks(np.arange(500, 2501, step=500))
        plt.gca().xaxis.set_major_locator(MultipleLocator(20))
        plt.yticks(fontsize = 12)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title(f'Boxplot of the spectral signature of class {lcz} obtained from PRISMA image')

        plt.show()
    
    interactive_plot = widgets.interact(boxplot_training_samples_widgets, lcz = lcz_class_w)
    
#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

def histogram_training_samples(image, shapes, legend, wvl_dict):
    
    # selection of the LCZ class
    lcz_class_w = widgets.RadioButtons(
        options=[value[0] for value in legend.values()],
        description='Select the LCZ class:',
        style = {'description_width': 'initial'},
        disabled=False)
    
    # selection of the band
    band_int_w = widgets.Dropdown(
        options = [round(value, 1) for value in wvl_dict.values()],
        description= 'Wavelength [nm]:',
        disabled = False,
        style = {'description_width': 'initial'}
    )
    
    def band_hist_plot(lcz, band):
        lcz_class = next(key for key, values in legend.items() if values[0] == lcz_class_w.value)
        band_threshold = 1e-8
        with rasterio.open(image) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes[lcz_class], crop=True, pad=True)
            out_image[out_image < band_threshold] = np.nan

        # store the reflectance values in a Pandas dataframe (each column is a band)
        n_bands = out_image.shape[0]
        flat_image_data = np.reshape(out_image, (n_bands, -1)).T

        df = pd.DataFrame(flat_image_data)
        df = df.rename(columns = wvl_dict)
        df = df.rename(columns={col: f'{round(col, 1):.1f}' for col in df.columns})
        # clear the previous plot
        plt.clf()
        fig, ax1 = pl.subplots(1, sharex = True, figsize=(10,8))
        sns.histplot(data = df, x = str(round(band_int_w.value,1)), kde = True, color = legend[lcz_class][1]).set(title=f'Wavelength: {str(round(band_int_w.value,1))} nm - Class {legend[lcz_class][0]}')

    interactive_plot = widgets.interact(band_hist_plot, lcz = lcz_class_w, band=band_int_w)
    
    
#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

from scipy.stats import linregress

def correlation_training_samples(image, shapes, legend, wvl_dict):
    
    # selection of the LCZ class
    lcz_class_w = widgets.RadioButtons(
        options=[value[0] for value in legend.values()],
        description='Select the LCZ class:',
        style = {'description_width': 'initial'},
        disabled=False)
    
    # selection of the bands
    band_w = widgets.SelectMultiple(
        options = [round(value, 1) for value in wvl_dict.values()],
        description= 'Wavelength [nm]:',
        rows = 10,
        disabled = False,
        style = {'description_width': 'initial'}
    )
    
    def correlation_training_samples_widgets(lcz, bands):
        
        lcz_class = next(key for key, values in legend.items() if values[0] == lcz)
        band_w_sel = band_w.value
        sel_wvls = list(band_w_sel)
        
        band_threshold = 1e-8
        with rasterio.open(image) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes[lcz_class], crop=True, pad=True)
            out_image[out_image < band_threshold] = np.nan

        # store the reflectance values in a Pandas dataframe (each column is a band)
        n_bands = out_image.shape[0]
        flat_image_data = np.reshape(out_image, (n_bands, -1)).T

        df = pd.DataFrame(flat_image_data)
        df = df.rename(columns = wvl_dict)
        df = df.rename(columns={col: f'{round(col, 1):.1f}' for col in df.columns})

        sel_wvls = [str(value) for value in sel_wvls]

        sel = df[sel_wvls]  #select some bands
        sel.dropna(inplace=True)

        corr_matrix = sel.corr()

        # create a heatmap of the correlation matrix
        sns.set(style="white")
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(corr_matrix, square=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, annot=True)
        plt.title("Correlation matrix")
        plt.show()
    
    interactive_plot = widgets.interact(correlation_training_samples_widgets, lcz = lcz_class_w, bands = band_w)
    

#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

def jeffries_matusita_distance(class1, class2):
    """
    Compute Jeffries-Matusita distance between two classes in a multispectral image.

    Parameters:
    - class1: 2D NumPy array representing the spectral signatures of class 1.
    - class2: 2D NumPy array representing the spectral signatures of class 2.

    Returns:
    - JM distance between the two classes.
    """
    
    # compute covariance matrix
    cov_class1 = np.cov(class1, rowvar=False)
    cov_class2 = np.cov(class2, rowvar=False)
    
    # compute mean difference
    mean_diff = np.mean(class1, axis=0) - np.mean(class2, axis=0)

    inv_cov_sum = np.linalg.inv((cov_class1 + cov_class2)/2)
  
    tmp = np.dot(mean_diff.T, inv_cov_sum)
    tmp = np.dot(tmp, mean_diff)
    MH = tmp
    
    tmp = np.linalg.det((cov_class1 + cov_class2)/2) / np.sqrt( np.linalg.det(cov_class1)*np.linalg.det(cov_class2) )
    tmp = np.log(tmp)
    B = MH/8.0 + tmp/2.0
    jm_distance = 2 * (1 - np.exp(-B))

    return jm_distance, mean_diff, cov_class1, cov_class2


#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

def spectral_separability(selected_classes, legend, sensor, pixels_spectral_sign, pixels_spectral_sign_s):

        # collect the pixels of the two classes inside two arrays, and set to float64
        # if PRISMA is selected, one band every 10 bands is extracted
        if sensor == 'PRISMA':
            class1_data = pixels_spectral_sign[selected_classes[0]]
            class2_data = pixels_spectral_sign[selected_classes[1]]
            class1_data = class1_data[:, ::10]
            class2_data = class2_data[:, ::10]
        elif sensor == 'Sentinel-2':
            class1_data = pixels_spectral_sign_s[selected_classes[0]]
            class2_data = pixels_spectral_sign_s[selected_classes[1]]

        class1_data = class1_data.astype('float64')
        class2_data = class2_data.astype('float64')
        
        # check if there is any nan inside the data
        has_nans_class1_data = np.isnan(class1_data).any()
        has_nans_class2_data = np.isnan(class2_data).any()

        if has_nans_class1_data:
            print("The first selected class contains NaN values!")
            return
        elif has_nans_class2_data:
            print("The second selected class contains NaN values!")
            return
        else:
            print("The two classes do not contain NaN values.")

        jm_distance, mean_diff, cov_class1, cov_class2 = jeffries_matusita_distance(class1_data, class2_data)
        print(f'Jeffries-Matusita distance among classes {legend[selected_classes[0]][0]} and {legend[selected_classes[1]][0]} using {sensor} is: {jm_distance}')
        
        return jm_distance, mean_diff, cov_class1, cov_class2

#------------------------------------------------#
# Function used in 3 - Plotting.ipynb

from itertools import combinations

def spectral_separability_all_classes(legend, sensor, sampling, pixels_spectral_sign, pixels_spectral_sign_s):
    
    results = {}
    lcz_classes = [key for key, value in legend.items()]
    index_names = [f"{legend[class1][0]} - {legend[class2][0]}" for class1, class2 in combinations(lcz_classes, 2)]
    
    # collect the pixels of the two classes inside two arrays, and set to float64
    # if PRISMA is selected, one band every 10 bands is extracted
    if sensor == 'PRISMA':
        pixels = pixels_spectral_sign
    else:
        pixels = pixels_spectral_sign_s
    
    for class1, class2 in combinations(lcz_classes, 2):
        class1_data = pixels[class1]
        class2_data = pixels[class2]
        
        if sampling == 'Bands closest to S2':
            class1_data = class1_data[:, positions]
            class2_data = class2_data[:, positions]
        elif sampling == '1 band every 10 bands':
            class1_data = class1_data[:, ::10]
            class2_data = class2_data[:, ::10]
            # exclude water vapor absorpion bands
            exclude_columns = [11, 16]
            columns_to_keep = [col for col in range(class1_data.shape[1]) if col not in exclude_columns]
            class1_data = class1_data[:, columns_to_keep]
            class2_data = class2_data[:, columns_to_keep]

        class1_data = class1_data.astype('float64')
        class2_data = class2_data.astype('float64')
        
        # check if there is any nan inside the data
        has_nans_class1_data = np.isnan(class1_data).any()
        has_nans_class2_data = np.isnan(class2_data).any()

        if has_nans_class1_data or has_nans_class2_data:
            print(f"The {class1} and {class2} classes contain NaN values using {sensor}. Skipping calculation.")
            continue

        jm_distance, mean_diff, cov_class1, cov_class2 = jeffries_matusita_distance(class1_data, class2_data)
        results[(class1, class2)] = {'jm_distance': jm_distance, 'mean_diff': mean_diff, 'cov_class1': cov_class1, 'cov_class2': cov_class2}
    
    # arrange the dictionary inside a dataframe
    # dataframe with jm_distance, mean_diff, cov_class1, cov_class2 for each couple of classes
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index, names=['Class 1', 'Class 2'])
    df.index = df.index.map(lambda x: '-'.join(map(str, x)))
    df.columns = ['jm_distance', 'mean_diff', 'cov_class1', 'cov_class2']
    
    # dataframe with only jm_distance for each couple of classes
    
    jm_distance_df = df[['jm_distance']].copy()
    jm_distance_df.index = jm_distance_df.index.map(lambda x: '-'.join(map(str, x)))
    jm_distance_df.columns = ['jm_distance']
    return results, df, jm_distance_df


#------------------------------------------------#
# Function used in 4 - PCA.ipynb

def prepare_input_pca(selected_prisma_image):
    
    with rasterio.open(selected_prisma_image) as src:
        original_image = src.read().astype(rasterio.float32)
        metadata = src.meta
    
    print('The selected PRISMA image has the following shape: ')
    print(f'Number of bands: {original_image.shape[0]}')
    print(f'Number of rows: {original_image.shape[1]}')
    print(f'Number of columns: {original_image.shape[2]}')
    
    band_threshold = 0.0000001
    original_image = original_image[~np.all(original_image <= band_threshold, axis=(1,2))]
    
    n_pixels = original_image.shape[1] * original_image.shape[2]
    n_bands = original_image.shape[0]
    image_flat = original_image.reshape(n_bands, n_pixels)
    
    image_flat_move = np.moveaxis(image_flat, -1, 0)
    
    input_image = image_flat_move

    
    return input_image, original_image, metadata, n_bands
    

#------------------------------------------------#
# Function used in 4 - PCA.ipynb

from sklearn.decomposition import PCA

def perform_pca(n_bands, selected_prisma_image, input_image):
    # Perform PCA on the reshaped array
    pca = PCA(n_components=n_bands)
    pca.fit(input_image)
    pc_transf = pca.transform(input_image)
    
    pc_transf_move = np.moveaxis(pc_transf, 0, -1)
    
    pc_transf_reshaped = pc_transf_move.reshape(n_bands, selected_prisma_image.shape[1], selected_prisma_image.shape[2])
    
    print('The computed PC matrix has the following shape: ')
    print(f'Number of PCs: {pc_transf_reshaped.shape[0]}')
    print(f'Number of rows: {pc_transf_reshaped.shape[1]}')
    print(f'Number of columns: {pc_transf_reshaped.shape[2]}')
    
    return pca, pc_transf, pc_transf_reshaped


#------------------------------------------------#
# Function used in 4 - PCA.ipynb

def plot_explained_var(x_bar_components, pca, cumulativeVar):
    
    # create a list of PC names
    pc_names = ['PC' + str(i) for i in range(1, x_bar_components+1)]
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=pc_names, y=pca.explained_variance_ratio_[0:x_bar_components], mode='markers+lines', name='Explained Variance Ratio', marker=dict(symbol='circle')))
    fig.add_trace(go.Scatter(x=pc_names, y=cumulativeVar[0:x_bar_components], mode='markers+lines', name='Cumulative Explained Variance Ratio', marker=dict(symbol='circle')))

    fig.update_layout(xaxis_title='Principal Component', yaxis_title='Variance Ratio')

    fig.update_xaxes(type='category')

    fig.show()
      

#------------------------------------------------#
# Function used in 4 - PCA.ipynb

def plot_loadings(loadings):
    
    PCx_w = widgets.IntSlider(value=1, min=1, max=234, description='PC on x-axis:', disabled=False, continuous_update=False)
    PCy_w = widgets.IntSlider(value=2, min=1, max=234, description='PC on y-axis:', disabled=False, continuous_update=False)
    PC_box = widgets.HBox([PCx_w, PCy_w])
    
    scale_w = widgets.RadioButtons(
        options = ['logarithmic', 'linear'],
        description = 'Select the plot scale: ',
        style = {'description_width': 'initial'}
    )
    
    def plot_loadings_widget(PCx, PCy, scale):  
        if scale == 'logarithmic':
            # Create scatter plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=loadings[:, PCx-1], y=loadings[:, PCy-1], mode='markers', marker = dict(color = 'red', size = 6), name = 'Loading'))

            labels = [f'B{i}' for i in range(1, len(loadings[PCx-1]) + 1)]
            fig.add_trace(go.Scatter(x=loadings[:, PCx-1], y=loadings[:, PCy-1], mode='text', text=labels, textposition='top right', name = 'Band number'))

            # Set figure size
            fig.update_layout(width=1000, height=800, title=f'Loadings of the PCs - logarithmic scale')

            # Set axis properties
            fig.update_layout(
                xaxis=dict(zeroline=True, showline=True, zerolinecolor = '#85A4BF', title=f'PC{PCx}', type = 'log'),
                yaxis=dict(zeroline=True, showline=True, zerolinecolor = '#85A4BF', title=f'PC{PCy}', type = 'log'),

            )
            fig.show()
        else:
            # Create scatter plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=loadings[:, PCx-1], y=loadings[:, PCy-1], mode='markers', marker = dict(color = 'red', size = 6), name = 'Loading'))

            labels = [f'B{i}' for i in range(1, len(loadings[PCx-1]) + 1)]
            fig.add_trace(go.Scatter(x=loadings[:, PCx-1], y=loadings[:, PCy-1], mode='text', text=labels, textposition='top right', name = 'Band number'))

            # Set figure size
            fig.update_layout(width=1000, height=600, title=f'Loadings of the PCs - logarithmic scale')

            # Set axis properties
            fig.update_layout(
                xaxis=dict(zeroline=True, showline=True, zerolinecolor = '#85A4BF', title=f'PC{PCx}'),
                yaxis=dict(zeroline=True, showline=True, zerolinecolor = '#85A4BF', title=f'PC{PCy}'),

            )
            fig.show()
        
    interactive_plot = widgets.interact(plot_loadings_widget, PCx = PCx_w, PCy = PCy_w, scale = scale_w)
    


#------------------------------------------------#
# Function used in 4 - PCA.ipynb    

def plot_loadings_2(loadings):
    
    PC_w = widgets.IntSlider(value=1, min=1, max=234, description='PC:', disabled=False, continuous_update=False)

    def plot_loadings_widget_2(PC):  
        # Create scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[i for i in range(1, len(loadings)+1)], y=loadings[:, PC-1], mode='markers', marker = dict(color = 'red', size = 6), name = f'Loading relative to PC{PC}'))

        # Set figure size
        fig.update_layout(width=1000, height=800, title=f'Loadings of PC{PC}')

        # Set axis properties
        fig.update_layout(
            xaxis=dict(zeroline=True, showline=True, zerolinecolor = '#85A4BF', title='Band number'),
            yaxis=dict(zeroline=True, showline=True, zerolinecolor = '#85A4BF', title='Loading'),

        )
        fig.show()
    
    interactive_plot2 = widgets.interact(plot_loadings_widget_2, PC = PC_w)

#------------------------------------------------#
# Function used in 4 - PCA.ipynb

def export_pc(pc_transf_reshaped, sel_pcs, metadata, out_path):
    
    # Create the folder/directory if it doesn't exist
    if not os.path.exists('PCs'):
        os.makedirs('PCs')
    
    # Update the metadata and export as GeoTIFF file
    dst_meta = metadata
    dst_meta['count'] = sel_pcs
    dst_meta['dtype'] = 'float32'
    
    with rasterio.open(out_path, 'w', **dst_meta) as dst:
        dst.write(pc_transf_reshaped[0:sel_pcs,:,:])

#------------------------------------------------#
# Function used in 4 - PCA.ipynb

def plot_pc(selected_prisma_image, pcs_path):
    
    #open the PRISMA image and store the metadata
    with rasterio.open(selected_prisma_image) as src:
        metadata = src.meta
    
    prisma_image = rasterio.open(selected_prisma_image)
    prisma_image_array = prisma_image.read()
    prisma_image_array = prisma_image_array.transpose(1, 2, 0)
    
    #create the mask
    empty_value = np.nan  #this is done because scikit learn cannot use nan
    mask_prisma = np.amax(prisma_image_array, axis=2).astype(float)
    mask_prisma[mask_prisma > 0] = 1
    mask_prisma[mask_prisma <= 0] = empty_value
    
    # Open the PCs
    with rasterio.open(pcs_path) as dataset:
        layers = dataset.read()
        num_bands = dataset.count
        
    # Apply the mask to every PC
    masked_layers = np.empty((num_bands, mask_prisma.shape[0], mask_prisma.shape[1]))
    for band in range(num_bands):
        layer = layers[band, :, :].squeeze()
        layer = layer[:mask_prisma.shape[0], :mask_prisma.shape[1]]
        masked_layer = layer * mask_prisma
        masked_layers[band, :, :] = masked_layer
        
    # Save the masked TIFF file
    out_path = pcs_path[:-8] + '_masked' + pcs_path[-8:]
    metadata['count'] = num_bands
    with rasterio.open(out_path, 'w', **metadata) as dst:
        dst.write(masked_layers)
    
    
    # Plot the PCs
    
    # first, set proper scalebar by finding the min and max values
    # along the PCs
    lowest_value = np.inf  # Initialize with a high value
    highest_value = -np.inf  # Initialize with a low value
    lowest_band = None
    highest_band = None
    for band in range(num_bands):
        value = np.nanmin(masked_layers[band, :, :])
        if value < lowest_value:
            lowest_value = value
            lowest_band = band
        value = np.nanmax(masked_layers[band, :, :])
        if value > highest_value:
            highest_value = value
            highest_band = band
    
    # then, plot the PCs
    
    pc_w = widgets.Dropdown(
        options = [i for i in range(1, num_bands+1)],
        description = 'Select the PC: ',
        disabled = False,
        style = {'description_width': 'initial'}
    )
    
    def plot_pc_widget(pc):
        plt.figure(figsize = (10,10))
        height, width = masked_layers[pc-1, :, :].shape
        extent = [0, width, height, 0]  # left, right, bottom, top
        plt.imshow(masked_layers[pc-1, :, :], cmap="Greens", vmin=np.nanquantile(masked_layers[lowest_band, :, :], 0.1), 
                   vmax=np.nanquantile(masked_layers[highest_band, :, :], 0.9), extent=extent)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Principal Component {pc}')

        plt.colorbar()
        
    interactive_plot = widgets.interact(plot_pc_widget, pc = pc_w)


#------------------------------------------------#
# Function used in 5 - Classification.ipynb

def training_area(sel_prisma_date, legend):
    
    vector_LCZ_path = './layers/training_samples/training_set_' + sel_prisma_date.replace('-', '') + '.gpkg'
    train_data = gpd.read_file(vector_LCZ_path)

    # Specify the column to plot
    column_names = 'LCZ'
    
    # Calculate the total area for each LCZ class to check if the training samples have balanced area. It is important to keep data balanced for the next classification steps (this is relevant expecially for urban classes, while natural classed usually are more easily classified):
    total_area = train_data.groupby(column_names)['geometry'].apply(lambda x: x.area.sum())
    
    # Check the list of LCZ classes available in the provided training set:
    train_data['LCZ_name'] = train_data['LCZ'].map(legend).str[0]
    classes_LCZ = list(train_data.LCZ_name.unique())
    classes_LCZ.sort()
    print("List of training samples LCZ classes: ", classes_LCZ)
    
    # Create a dictionary containing the class numbers and the desiderd color to be used for plotting
    cmap_colors = [legend[key][1] for key in legend.keys()]
    cmap = plt.cm.colors.ListedColormap(cmap_colors, name='LCZ classes colormap')
    

    # Create the bar trace
    total_area.index = total_area.index.map(lambda x: legend[x][0])
    bar_trace = go.Bar(
        x=total_area.index.astype(str),
        y=total_area,
        marker=dict(color=cmap_colors),
    )

    # Create the layout
    layout = go.Layout(
        title='Area of the training samples',
        xaxis=dict(title='Class'),
        yaxis=dict(title='Total area [m²]', tickformat='1.1e'),
        height = 500,
        width = 600
    )

    # Create the figure and add the trace
    fig = go.Figure(data=[bar_trace], layout=layout)

    # Display the figure
    fig.show()
    
    return vector_LCZ_path

#------------------------------------------------#
# Function used in 5 - Classification.ipynb

from osgeo import gdal, ogr, gdalconst, gdal_array, osr


def rasterize_training(raster_reference, shp, output, attribute, projection):
    
    """
    Rasterizes a shapefile/geopackage (training vector) onto a raster reference image.

    Args:
        raster_reference (str): Path to the raster reference image.
        shp (str): Path to the shapefile.
        output (str): Path to the output rasterized image.
        attribute (str): Attribute field to use for rasterization.
        projection (int, optional): EPSG code for the desired projection. Defaults to 32632.

    Returns:
        None
    """
    
    data = gdal.Open(raster_reference, gdalconst.GA_ReadOnly)
    geo_transform = data.Getgeo_transform()
    prj=data.GetProjection()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    #print(x_min,y_max, x_max, y_min, x_res, y_res)
    mb_v = ogr.Open(shp)
    mb_l = mb_v.GetLayer()
    pixel_width = geo_transform[1]
    target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.Setgeo_transform((x_min, pixel_width, 0, y_max, 0, -pixel_width))  # minus sign because opposite convention
    # Set the spatial reference of the target dataset to match the raster
    target_ds.SetProjection(data.GetProjection())
    band = target_ds.GetRasterBand(1)
    NoData_value = 0
    band.Fill(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE="+attribute])
    #adding a spatial reference
    new_rasterSRS = osr.SpatialReference()
    new_rasterSRS.ImportFromEPSG(projection)
    target_ds.SetProjection(new_rasterSRS.ExportToWkt())

    target_ds = None
    print(f"The file has been rasterized!")

    
    
#------------------------------------------------#
# Function used in 5 - Classification.ipynb

import matplotlib.colors as colors
from rasterio.plot import show

def plot_raster_training(raster, legend):
    
    with rasterio.open(raster) as src:
        # Read the raster data
        rasterized_result = src.read()
        rasterized_result[np.isnan(rasterized_result)] = 0
        # Convert the data to integer type
        rasterized_result = rasterized_result.astype(np.float32)
        rasterized_result[rasterized_result <= 0] = np.nan
        #print("Classes: ", np.unique(rasterized_result))
        
        bounds = [int(key) for key in legend.keys()]
    bounds.append(bounds[-1]+1)
    
    cmap_colors = [legend[key][1] for key in legend.keys()]
    cmap = plt.cm.colors.ListedColormap(cmap_colors, name='LCZ classes colormap')
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Plot the raster data with a custom figure size
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.title("Rasterized training samples")
    show(rasterized_result, cmap=cmap, ax=ax, interpolation='none', norm=norm);
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a legend with the correct class colors
    labels = [legend[i][0] for i in legend.keys()]
    handles = [plt.Rectangle((0, 0), 1, 1, color=legend[key][1]) for key in list(legend.keys())]
    legend = plt.legend(handles, labels, title='Class', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = 12)
    plt.setp(legend.get_title(), fontsize='12')  # Adjust the font size of the legend title

    # Adjust the plot layout to accommodate the legend outside
    ##plt.tight_layout(rect=[0, 0, 0.85, 1])  # Increase the left margin to make space for the legend

    plt.show()


#------------------------------------------------#
# Function used in 5 - Classification.ipynb

from shapely.geometry import box

def prisma_bbox(selected_prisma_image, sel_prisma_date):
    
    with rasterio.open(selected_prisma_image) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
    
    print(raster_bounds)
    xmin, ymin, xmax, ymax = raster_bounds
    bbox_polygon = box(xmin, ymin, xmax, ymax)

    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_polygon], crs=raster_crs)

    # Create the folder/directory if it doesn't exist
    if not os.path.exists('study_area'):
        os.makedirs('study_area')

    bbox = 'study_area/study_area_' + sel_prisma_date.replace('-', '')+'.gpkg'
    bbox_gdf.to_file(bbox)
    print(f'The bounding box has been generated and saved in {bbox}.')
    
    return bbox


#------------------------------------------------#
# Function used in 5 - Classification.ipynb

def mask_prisma_image(selected_prisma_image):
    
    prisma_image = rasterio.open(selected_prisma_image)
    prisma_image_array = prisma_image.read()
    prisma_image_array = prisma_image_array.transpose(1, 2, 0)

    # Mask for coregistered imagery
    empty_value = 0  #this is done because scikit learn cannot use nan
    mask_prisma = np.amax(prisma_image_array, axis=2).astype(float)
    mask_prisma[mask_prisma > 0] = 1
    mask_prisma[mask_prisma <= 0] = empty_value
    print(f"PRISMA mask shape --> {mask_prisma.shape}")
    
    plt.imshow(mask_prisma, cmap = 'binary')
    plt.colorbar(cmap = plt.get_cmap('binary'), ticks=[0, 1])
    plt.xticks([])
    plt.yticks([])
    plt.title("PRISMA mask")
    
    return mask_prisma


#------------------------------------------------#
# Function used in 5 - Classification.ipynb

def open_layer(layer_path, mask_prisma):
    with rasterio.open(layer_path) as dataset:
        layer = dataset.read().squeeze()
    
    layer = layer[:mask_prisma.shape[0], :mask_prisma.shape[1]]
    layer[layer < 0] = 0
    #layer = np.nan_to_num(layer, nan=0)
    min_val = np.min(layer)
    max_val = np.max(layer)

    if max_val > 1: 
        normalized_array = (layer - min_val) / (max_val - min_val)
        print(f"{layer_path} shape: {layer.shape} ---> Max value: {np.max(normalized_array):.2f} | Min value: {np.min(normalized_array):.2f}")
        return normalized_array * mask_prisma
    else:
        print(f"{layer_path} shape: {layer.shape} ---> Max value: {np.max(layer):.2f} | Min value: {np.min(layer):.2f}")
        return layer  * mask_prisma

    
    
#------------------------------------------------#
# Function used in 5 - Classification.ipynb

def plot_ucl(imperv, perc_build, svf, canopy_height, buildings):
    

    # Display the classification layers
    fig, axs = plt.subplots(3, 2, figsize=(16, 16))

    im1 = axs[0, 0].imshow(svf)
    axs[0, 0].set_title('Sky View Factor [0-1]')
    cbar1 = fig.colorbar(im1, ax=axs[0, 0])

    im2 = axs[0, 1].imshow(imperv)
    axs[0, 1].set_title('Impervious Surface Fraction [0-1]')
    cbar2 = fig.colorbar(im2, ax=axs[0, 1])

    im3 = axs[1, 0].imshow(perc_build)
    axs[1, 0].set_title('Building Surface Fraction [0-1]')
    cbar3 = fig.colorbar(im3, ax=axs[1, 0])

    im4 = axs[1, 1].imshow(canopy_height)
    axs[1, 1].set_title('Tree Canopy Height [0-1]')
    cbar4 = fig.colorbar(im4, ax=axs[1, 1])

    im5 = axs[2, 0].imshow(buildings, vmax=0.2)
    axs[2, 0].set_title('Buildings [0-1]')
    cbar5 = fig.colorbar(im5, ax=axs[2, 0])

    # Remove the axis for the blank subplot
    axs[2, 1].axis('off')
    
    # Remove x and y ticks from every subplot
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ax.get_title(), fontsize=14)

    # Adjust the spacing between subplots
    plt.tight_layout()
    

    plt.show()

    
#------------------------------------------------#
# Function used in 5 - Classification.ipynb

def clip_training_sample(image_path, training_path, sel_prisma_date, study_area):
    
    # Read in our image and training set image
    img_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    roi_ds = training_path

    # Create layers for the image and training samples - The image corresponds to the image with PCs
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    # Training samples
    roi_new_path = './layers/training_samples/training_set_'+ sel_prisma_date.replace('-', '')+'_30m.tif'
    clip_image_study_area(roi_ds, roi_new_path, study_area)

    with rasterio.open(roi_new_path) as roi_src:
        roi = roi_src.read().squeeze()

    print("Classes in the ROI: ", np.unique(roi))
    
    return img, roi


#------------------------------------------------#
# Function used in 5 - Classification.ipynb

def check_layers_dimension(imperv, perc_build, svf, canopy_height, buildings, roi, img):
    
    # Check dimensions and compute new layers if necessary
    array_dim = 3 # if they already have dimension 3 don't expand dimensions

    if imperv.ndim < array_dim:
        imperv = np.expand_dims(imperv, axis=-1)
    if perc_build.ndim < array_dim:
        perc_build = np.expand_dims(perc_build, axis=-1)
    if svf.ndim < array_dim:
        svf = np.expand_dims(svf, axis=-1)
    if canopy_height.ndim < array_dim:
        canopy_height = np.expand_dims(canopy_height, axis=-1)
    if buildings.ndim < array_dim:
        buildings = np.expand_dims(buildings, axis=-1)


    print("Impervious shape: ", imperv.shape)
    print("Build percentage shape: ", perc_build.shape)
    print("SVF shape: ", svf.shape)
    print("Tree Canopy Height shape: ", canopy_height.shape)
    print("Building shape: ", buildings.shape)
    print("ROI shape: ", roi.shape)
    print("PRISMA PC image: ", img.shape)
    
    
    # Calculate the difference in width between the current shape and the target shape
    width_diff = np.abs(imperv.shape[0] - roi.shape[0])
    height_diff = np.abs(imperv.shape[1] - roi.shape[1])
    print(roi.shape)
    print(width_diff, height_diff)

    # Pad the array with zeros along the width
    roi = np.pad(roi, ((0, 0), (0, width_diff)), mode='constant')
    roi = np.pad(roi, ((0, 0), (0, height_diff)), mode='constant')
    print(f"The ROI shape is --> {roi.shape}")

    return imperv, perc_build, svf, canopy_height, buildings, roi


#------------------------------------------------#
# Function used in 5 - Classification.ipynb

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
#import xgboost as xgb

def parameter_tuning(classification_method, X_train, y_train):
    
    if classification_method == 'RandomForest':
        # Create a Random Forest classifier object
        cl = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 150], #[100, 150, 200, 500]
            'max_features': ['auto', 'sqrt'], #['auto', 'sqrt', 'log2']
            'criterion': ['gini'] #['gini', 'entropy']
        }
        print('Using Random Forest')

    elif classification_method == 'AdaBoost':
        # Create an AdaBoost classifier object
        cl = AdaBoostClassifier()
        param_grid = {
            'n_estimators': [100, 150, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'algorithm': ['SAMME', 'SAMME.R']
        }
        print('Using AdaBoost')

    elif classification_method == 'GradientBoost':
        # Create a Gradient Boosting classifier object
        cl = GradientBoostingClassifier()
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_features': ['log2', 'sqrt', 'log2'],
            'criterion': ['friedman_mse', 'mae', 'mse'],
            'n_estimators': [100, 150, 200, 500]
        }
        print('Using Gradient Boost')

    elif classification_method == 'XGBoost':
        # Create an XGBoost classifier object
        cl = xgb.XGBClassifier()
        param_grid = {
            'n_estimators': [100, 150, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2]
        }
        print('Using XGBoost')
    
    
    # Fit the GridSearchCV object to the training data
    if classification_method == 'XGBoost':

        # Encode the target variable
        label_encoder = LabelEncoder()
        encoded_y = label_encoder.fit_transform(y_train)

        # Create a GridSearchCV object to find the best hyperparameters
        grid_search = GridSearchCV(cl, param_grid, scoring='accuracy', cv=5, verbose=10)
        grid_search.fit(X_train, encoded_y)

        # Get the best classifier
        best_params = grid_search.best_estimator_
        print('Best Classifier:', best_params)

    else:
        # Create a GridSearchCV object to find the best hyperparameters
        grid_search = GridSearchCV(cl, param_grid, scoring='accuracy', cv=5, verbose=10)
        # Fit the GridSearchCV object to the training data
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("Best hyperparameters:", best_params)
    
    return best_params


#------------------------------------------------#
# Function used in 5 - Classification.ipynb

def classification(classification_method, best_params, X_train, y_train, X_test):
    
    # Create classifier objects based on the selected classification method
    if classification_method == 'RandomForest':
        # Create a Random Forest classifier object
        clc = RandomForestClassifier(max_features=best_params['max_features'], 
                                     n_estimators=best_params['n_estimators'], 
                                     criterion=best_params['criterion'],
                                     oob_score=True)

    elif classification_method == 'AdaBoost':
        # Create an AdaBoost classifier object
        clc = AdaBoostClassifier(n_estimators=best_params['n_estimators'], 
                                 learning_rate=best_params['learning_rate'])

    elif classification_method == 'GradientBoost':
        # Create a Gradient Boosting classifier object
        clc = GradientBoostingClassifier(learning_rate=best_params['learning_rate'], 
                                         max_features=best_params['max_features'], 
                                         criterion=best_params['criterion'],
                                         n_estimators=best_params['n_estimators'])

    elif classification_method == 'XGBoost':
        # Create an XGBoost classifier object
        clc = xgb.XGBClassifier(n_estimators=best_params.n_estimators,
                                    learning_rate=best_params.learning_rate,
                                    max_depth=best_params.max_depth)
    
    # Use the best model to make predictions on the test set
    # If XGB
    if classification_method == 'XGBoost':
        # Encode the target variable
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        clc.fit(X_train, y_train)

        y_pred = clc.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred)

    # If RF, AB, GB
    else:
        clc.fit(X_train, y_train)
        y_pred = clc.predict(X_test)
        
    return y_pred, clc

#------------------------------------------------#
# Function used in 5 - Classification.ipynb and 6 - Validation.ipynb 

# Define a function to print selected model metrics
def print_metrics(y_true, y_pred):
    '''
    Print accuracy score, confusion matrix, and classification report.
    
    Args:
    - y_true: ground-truth labels
    - y_pred: predicted labels
    '''
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    # Plot confusion matrix as a heatmap
    labels = sorted(set(y_true))
    matrix = pd.DataFrame(confusion, index=labels, columns=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    
    
    print('-------')
    print(f'Accuracy: {accuracy:.2f}')
    print('-------')
    print('Classification Report:')
    print(report)


#------------------------------------------------#
# Function used in 5 - Classification.ipynb

from scipy.ndimage import median_filter

def export_classified_map(img, clc, X, mask_prisma, selected_prisma_image, classification_method, sel_prisma_date):
    
    img = np.nan_to_num(img)
    # If XGB
    if classification_method == 'XGBoost':
        print('Using XGBoost')
        # Reshape the input image into a long 2D array for classification
        new_shape = (img.shape[0] * img.shape[1], img.shape[2])

        img_as_array = img[:, :, :X.shape[1]].reshape(new_shape)
        print('Reshaped from {o} to {n}'.format(o=img.shape,
                                                n=img_as_array.shape))

        # Encode the target variable using LabelEncoder
        label_encoder = LabelEncoder()
        encoded_y = label_encoder.fit_transform(y)

        # Fit the XGBoost classifier on the encoded data
        clc.fit(X, encoded_y)

        # Predict class labels for each pixel
        class_prediction = clc.predict(img_as_array)

        # Reshape the classification map to match the image shape
        class_prediction = class_prediction.reshape(img[:, :, 0].shape)

         # Flatten and decode the predicted labels back to original values
        reshaped_prediction = class_prediction.flatten()
        class_prediction = label_encoder.inverse_transform(reshaped_prediction)
        class_prediction = class_prediction.reshape(img[:, :, 0].shape)

    # If RF, AB, GB
    else:
        print(f'Using {classification_method}')
        # reshape into long 2d array (nrow * ncol, nband) for classification
        new_shape = (img.shape[0] * img.shape[1], img.shape[2])

        img_as_array = img[:, :, :X.shape[1]].reshape(new_shape)
        print('Reshaped from {o} to {n}'.format(o=img.shape,
                                                n=img_as_array.shape))

        # Now predict for each pixel
        class_prediction = clc.predict(img_as_array)

        # Reshape our classification map
        class_prediction = class_prediction.reshape(img[:, :, 0].shape)
        
        
    # Create PRISMA mask
    mask_prisma_export = mask_prisma
    mask_prisma_export[mask_prisma == 0] = np.nan
    #plt.imshow(mask_prisma_export)
    #plt.title("PRISMA Mask")
    #plt.colorbar()
    
    # Save the images in GeoTIFF format
    prisma_image = rasterio.open(selected_prisma_image)
    kwargs = prisma_image.meta
    kwargs.update(
        dtype=rasterio.float32,
        nodata = np.nan,
        count=1)
    
    folder_path = "classified_images"

    # Check if the folder doesn't exist
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")
        
    # save the classified image with rasterio
    with rasterio.open(folder_path + '/classified_' + classification_method + '_' + sel_prisma_date.replace('-', '') + '_30m.tif', 'w', **kwargs) as dst:
        dst.write(class_prediction*mask_prisma_export, 1)
        print(f"The classified file {'/classified_' + classification_method + '_' + sel_prisma_date.replace('-', '') + '_30m.tif'} has been created!")
    
    
    # Application of the median filter
    print(f"Application of a median filter of size 3...")
    # define the size of the median filter window
    filter_size = 3
    # apply the median filter to the classified image
    smoothed_image = median_filter(class_prediction, size=(filter_size, filter_size))
    
    # save the classified image with rasterio
    with rasterio.open(folder_path + '/classified_' + classification_method + '_' + sel_prisma_date.replace('-', '') + '_medianfilter_30m.tif', 'w', **kwargs) as dst:
        dst.write(smoothed_image*mask_prisma_export, 1)
        print(f"The smoothed classified file {'/classified_' + classification_method + '_' + sel_prisma_date.replace('-', '') + '_medianfilter_30m.tif'} has been created!")
