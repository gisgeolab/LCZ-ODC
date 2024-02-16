import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.plot import show_hist
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import ipywidgets as widgets
from sklearn.decomposition import PCA
import pandas as pd
import geopandas as gpd
import pyproj
from shapely.geometry import Polygon
from shapely.geometry import box
import cv2

from ipyleaflet import Map, basemaps, basemap_to_tiles, DrawControl, LayersControl  #if error run: jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-leaflet


# -------------------------------------------------

def get_prisma_extent(prisma_path, sel_prisma_date):
    """
    Function to draw the extent of the selected PRISMA image in the interactive map.
    """
    PRISMA_image_path = prisma_path + 'PR_' + sel_prisma_date.replace('-', '') + '_30m.tif'
    
    with rio.open(PRISMA_image_path) as src:
        src_transform = src.transform
        src_height = src.height
        src_width = src.width
    
    x1, y1 = src_transform * (0, 0)
    x2, y2 = src_transform * (src_width, 0)
    x3, y3 = src_transform * (0, src_height)
    x4, y4 = src_transform * (src_width, src_height)
    
    # convert from WGS84/UTM32N to WGS84 (geographic coordinates)
    source_proj = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')
    target_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    
    x1_wgs84, y1_wgs84 = pyproj.transform(source_proj, target_proj, x1, y1)
    x2_wgs84, y2_wgs84 = pyproj.transform(source_proj, target_proj, x2, y2)
    x3_wgs84, y3_wgs84 = pyproj.transform(source_proj, target_proj, x3, y3)
    x4_wgs84, y4_wgs84 = pyproj.transform(source_proj, target_proj, x4, y4)
    
    df = pd.DataFrame({'lat': [y1_wgs84, y2_wgs84, y3_wgs84, y4_wgs84],
                    'long': [x1_wgs84, x2_wgs84, x3_wgs84, x4_wgs84]})
    
    return df

# -----------------------------------------------

def get_prisma_s2_wvl(prisma_meta, s2_meta):
    
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
    
    # Retrieve S2 central wavelengths
    central_wvl = s2_meta.getElementsByTagName('CENTRAL')
    wvl_s = []
    for elem in central_wvl:
        w = elem.firstChild.data
        w = np.float32(w)
        wvl_s.append(w)
    positions = [0, 7, 9, 10]
    positions.sort(reverse = True)
    for pos in positions:
        del wvl_s[pos]
    
    return wvl, wvl_dict_sorted, wvl_s

from matplotlib.ticker import MultipleLocator


# -------------------------------------------------

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



# -------------------------------------------------

def draw_map(center_lat, center_lon, zoom_level):
    """
    Function to draw an ipyleaflet map and interact with it. It is possible to get the coordinates values from the dc variable. Two basemaps are available.
    """
    satellite = basemap_to_tiles(basemaps.Gaode.Satellite)
    osm = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)

    prisma_map = Map(layers=(satellite, osm ), center=(center_lat, center_lon), zoom=zoom_level)

    dc = DrawControl()
    lc = LayersControl(position='topright')

    dc = DrawControl(
        marker={"shapeOptions": {"color": "#0000FF"}},
        circlemarker={}, polyline={}, polygon={}
    )

    def handle_draw(target, action, geo_json):
        print(action)
        print(geo_json)

    dc.on_draw(handle_draw)
    prisma_map.add_control(dc)
    prisma_map.add_control(lc)
    
    
    return prisma_map, dc

# -------------------------------------------------

def draw_aoi(coords, side):
    """
    Function to draw the area of interest in the map drawn with the draw_map function.
    """
    lat = coords[1]
    long = coords[0]
    
    p = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')
    east, north = p(long, lat)
    east = round(east)
    north = round(north)
    
    east_list = [east-side, east+side, east+side, east-side]
    north_list = [north+side, north+side, north-side, north-side]
    
    df_bb = pd.DataFrame(list(zip(east_list, north_list)), columns =['east', 'north'])
    
    polygon_geom = Polygon(zip(df_bb.east, df_bb.north))
    polygon = gpd.GeoDataFrame(index = [0], crs='epsg:32632', geometry = [polygon_geom])
    shapes = polygon.geometry
    
    p2 = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')
    long1, lat1 = p2(df_bb.iloc[:,0], df_bb.iloc[:,1], inverse=True)
    
    return lat1, long1, polygon

# -------------------------------------------------

def clip_pan_hs(hs_full, pan_full, polygon, hs_path, pan_path):
    """
    Function to clip the panchromatic and VNIR bands to the selected AOI.
    """
    
    # Define the AOI
    shapes = polygon.geometry
    with rio.open(hs_full) as src:
        out_meta = src.meta.copy()
        out_image, out_transform = mask(src, shapes, crop=True, all_touched=True)
    
    out_meta.update({"driver": "GTiff",
                      "height": out_image.shape[1],
                      "width": out_image.shape[2],
                      "transform": out_transform})
    
    with rio.open(hs_path, "w", **out_meta) as dest:
        dest.write(out_image)
        
    # Open the raster to be clipped
    with rio.open(pan_full) as src:
    # Get the bounds of the raster to be used for clipping
        with rio.open(hs_path) as mask_src:
            mask_bounds = mask_src.bounds
        # Create a bounding box geometry from the mask bounds
        mask_box = box(*mask_bounds)
        print(mask_box)
        # Clip the raster using the bounding box of the mask raster
        out_image, out_transform = mask(src, [mask_box], crop=True)
        out_meta = src.meta.copy()
        # Update the metadata with the new dimensions and transform
        out_meta.update({'height': out_image.shape[1],
                         'width': out_image.shape[2],
                         'transform': out_transform})
        # Write the clipped raster to a new file
        with rio.open(pan_path, 'w', **out_meta) as dest:
            dest.write(out_image)
    print("Clipping has been performed and images have been exported.")


# -------------------------------------------------

def resample_image(image, src_meta, dst_meta, dst_path, resampling_method):
    
    """Upsamples a multiband image to the same resolution of a target image using the selected resampling method
    and saves the upsampled image to the desired path
    
    Parameters:
    - image: image to be upsampled
    - src_meta: metadata of the image to be upsampled (source metadata)
    - dst_meta: metadata of the upsampled image (destination metadata)
    - dst_path: path where the upsampled image will be saved
    - resampling_method: upsampling method
    
    Returns:
    - upsampled_image: upsampled image
    """
    upsampled_image = np.zeros((dst_meta['count'], dst_meta['height'], dst_meta['width']), dtype=image.dtype)
    reproject(image, upsampled_image, src_transform=src_meta['transform'], src_crs=src_meta['crs'],
              dst_transform=dst_meta['transform'], dst_crs=dst_meta['crs'], resampling=resampling_method);
    
    with rio.open(dst_path, 'w', **dst_meta) as dst:
        dst.write(upsampled_image)
        
    return upsampled_image


#-----------------------------------------

def convert_to_RGB(prisma_path, image_pansh, image_hs, dst_meta, idx_red=32, idx_green=22, idx_blue=11):
    
    # Extract the corresponding bands
    red_pansh = image_pansh[idx_red-1, :, :]
    green_pansh = image_pansh[idx_green-1, :, :]
    blue_pansh = image_pansh[idx_blue-1, :, :]
    red_hs = image_hs[idx_red-1, :, :]
    green_hs = image_hs[idx_green-1, :, :]
    blue_hs = image_hs[idx_blue-1, :, :]
    
    # Save the RGB images to tif files
    with rio.open(prisma_path + 'validation/RGB_pansharpened.tif', 'w', **dst_meta) as dst:
        # Write the RGB data to the raster file
        dst.write(red_pansh.astype(rio.float32), 1)
        dst.write(green_pansh.astype(rio.float32), 2)
        dst.write(blue_pansh.astype(rio.float32), 3)
    with rio.open(prisma_path + 'validation/RGB_hs.tif', 'w', **dst_meta) as dst:
        # Write the RGB data to the raster file
        dst.write(red_hs.astype(rio.float32), 1)
        dst.write(green_hs.astype(rio.float32), 2)
        dst.write(blue_hs.astype(rio.float32), 3)

    print("RGB images has been saved.")
    
    # Open again the saved RGB images
    with rio.open(prisma_path + 'validation/RGB_pansharpened.tif') as src:
        # Read the data from the red, green, and blue bands
        data_pansh = src.read()
        src_meta = src.meta
    with rio.open(prisma_path + 'validation/RGB_hs.tif') as src:
        # Read the data from the red, green, and blue bands
        data_hs = src.read()
        src_meta = src.meta
    
    # Rearrange arrays shapes
    data_pansh = np.moveaxis(data_pansh, 0, -1)
    data_hs = np.moveaxis(data_hs, 0, -1)
    
    # Rearrange the data to increase contrast
    data_pansh[data_pansh<0] = 0
    data_pansh[:, :, 0] = (data_pansh[:, :, 0] - data_pansh[:, :, 0].min())/(data_pansh[:, :, 0].max() - data_pansh[:, :, 0].min())
    data_pansh[:, :, 1] = (data_pansh[:, :, 1] - data_pansh[:, :, 1].min())/(data_pansh[:, :, 1].max() - data_pansh[:, :, 1].min())
    data_pansh[:, :, 2] = (data_pansh[:, :, 2] - data_pansh[:, :, 2].min())/(data_pansh[:, :, 2].max() - data_pansh[:, :, 2].min())
    data_hs[data_hs<0] = 0
    data_hs[:, :, 0] = (data_hs[:, :, 0] - data_hs[:, :, 0].min())/(data_hs[:, :, 0].max() - data_hs[:, :, 0].min())
    data_hs[:, :, 1] = (data_hs[:, :, 1] - data_hs[:, :, 1].min())/(data_hs[:, :, 1].max() - data_hs[:, :, 1].min())
    data_hs[:, :, 2] = (data_hs[:, :, 2] - data_hs[:, :, 2].min())/(data_hs[:, :, 2].max() - data_hs[:, :, 2].min())
    
    
    print("Images are ready to be exported as PNG files.")
    
    return data_pansh, data_hs

#-----------------------------------------

def convert_to_RGB_s2(prisma_path, image_pansh, image_hs, dst_meta, idx_red=3, idx_green=2, idx_blue=1):
    
    # Extract the corresponding bands
    red_pansh = image_pansh[idx_red-1, :, :]
    green_pansh = image_pansh[idx_green-1, :, :]
    blue_pansh = image_pansh[idx_blue-1, :, :]
    red_hs = image_hs[idx_red-1, :, :]
    green_hs = image_hs[idx_green-1, :, :]
    blue_hs = image_hs[idx_blue-1, :, :]
    
    # Save the RGB images to tif files
    with rio.open(prisma_path + 'validation/s2_RGB_pansharpened.tif', 'w', **dst_meta) as dst:
        # Write the RGB data to the raster file
        dst.write(red_pansh.astype(rio.float32), 1)
        dst.write(green_pansh.astype(rio.float32), 2)
        dst.write(blue_pansh.astype(rio.float32), 3)
    with rio.open(prisma_path + 'validation/s2_RGB_hs.tif', 'w', **dst_meta) as dst:
        # Write the RGB data to the raster file
        dst.write(red_hs.astype(rio.float32), 1)
        dst.write(green_hs.astype(rio.float32), 2)
        dst.write(blue_hs.astype(rio.float32), 3)

    print("RGB images has been saved.")
    
    # Open again the saved RGB images
    with rio.open(prisma_path + 'validation/s2_RGB_pansharpened.tif') as src:
        # Read the data from the red, green, and blue bands
        data_pansh = src.read()
        src_meta = src.meta
    with rio.open(prisma_path + 'validation/s2_RGB_hs.tif') as src:
        # Read the data from the red, green, and blue bands
        data_hs = src.read()
        src_meta = src.meta
    
    # Rearrange arrays shapes
    data_pansh = np.moveaxis(data_pansh, 0, -1)
    data_hs = np.moveaxis(data_hs, 0, -1)
    
    # Rearrange the data to increase contrast
    data_pansh[data_pansh<0] = 0
    data_pansh[:, :, 0] = (data_pansh[:, :, 0] - data_pansh[:, :, 0].min())/(data_pansh[:, :, 0].max() - data_pansh[:, :, 0].min())
    data_pansh[:, :, 1] = (data_pansh[:, :, 1] - data_pansh[:, :, 1].min())/(data_pansh[:, :, 1].max() - data_pansh[:, :, 1].min())
    data_pansh[:, :, 2] = (data_pansh[:, :, 2] - data_pansh[:, :, 2].min())/(data_pansh[:, :, 2].max() - data_pansh[:, :, 2].min())
    data_hs[data_hs<0] = 0
    data_hs[:, :, 0] = (data_hs[:, :, 0] - data_hs[:, :, 0].min())/(data_hs[:, :, 0].max() - data_hs[:, :, 0].min())
    data_hs[:, :, 1] = (data_hs[:, :, 1] - data_hs[:, :, 1].min())/(data_hs[:, :, 1].max() - data_hs[:, :, 1].min())
    data_hs[:, :, 2] = (data_hs[:, :, 2] - data_hs[:, :, 2].min())/(data_hs[:, :, 2].max() - data_hs[:, :, 2].min())
    
    
    print("Images are ready to be exported as PNG files.")
    
    return data_pansh, data_hs

#------------------------------------------------#

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
    spectral_sign = {}
    spectral_sign_std = {}
    with rio.open(image) as src:
        for LCZ in LCZ_class:
            print(f'Computed spectral signature in the training samples for class: {legend[LCZ][0]}')
            out_image, out_transform = rio.mask.mask(dataset=src, shapes=shapes[LCZ], crop=True, pad=True)
            out_image[out_image == 0] = np.nan
        
            spectral_sign[LCZ] = np.nanmedian(out_image, axis=(1, 2))
            spectral_sign_std[LCZ] = np.nanstd(out_image, axis=(1, 2))
        
            spectral_sign[LCZ] = spectral_sign[LCZ][abs(spectral_sign[LCZ])>band_threshold] #remove values equal to zero
            spectral_sign_std[LCZ] = spectral_sign_std[LCZ][abs(spectral_sign_std[LCZ])>band_threshold] #remove values equal to zero
            
    return spectral_sign, spectral_sign_std


#------------------------------------------------#

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