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

def pan_pca(pan_data, hs_5m_data):
    
    """Apply principal component analysis (PCA) to hyperspectral image data and pansharpen it using panchromatic data.
    
    Parameters:
    - pan_data (ndarray): Panchromatic image data as a 2D array.
    - hs_5m_data (ndarray): Hyperspectral image data as a 3D array with shape (n_bands, n_rows, n_cols).
    
    Returns:
    - pc_reshaped (ndarray): Pansharpened hyperspectral image data as a 3D array with shape (n_bands, n_rows, n_cols).
    - variance_ratios (ndarray): Explained variance ratios of the PCA components.
    - pc_comp (ndarray): PCA components.
    """
    
    # Initialize PCA with the desired number of components
    print("Using PCA Pansharpening method")
    print(f"Pan data (5m resolution) shape is: {pan_data.shape}")
    print(f"Upsampled HS data (5m resolution) shape is: {hs_5m_data.shape}")
    print("---------------------")
    
    n_bands = hs_5m_data.shape[0]
    n_pixels = hs_5m_data.shape[1] * hs_5m_data.shape[2]
    
    image_flat = hs_5m_data.reshape(n_bands, n_pixels)
    image_flat_move = np.moveaxis(image_flat, -1, 0)
    image_flat_move.shape
    
    
    pca = PCA(n_components=n_bands)
    
    # Fit PCA to the input image data
    pca.fit(image_flat_move)
    
    # Transform the input image data using the learned PCA components
    pc = pca.transform(image_flat_move)
    
    # Extract the first principal component (I)
    I = pc[:, 0]
    
    # Flatten the panchromatic data into a 1D array
    pan_flat = pan_data.reshape(1, pan_data.shape[1]*pan_data.shape[2])
    
    # Move the axis to the first dimension
    pan_flat = np.moveaxis(pan_flat, -1, 0)
    print(f"Flattened pan data shape is: {pan_flat.shape}")
    
    # Standardize the panchromatic data using the mean and standard deviation of I
    pan_flat_st = (pan_flat - np.mean(pan_flat)) * np.std(I, ddof=1) / np.std(pan_flat, ddof=1) + np.mean(I)
    
    # Replace the first principal component of pc with the standardized panchromatic data
    pc[:, 0] = pan_flat_st.flatten()
    
    # Invert the PCA transformation to reconstruct the image data
    pc_transformed = pca.inverse_transform(pc)
    
    # Move the first axis back to the last dimension
    pc_transformed = np.moveaxis(pc_transformed, -1, 0)
    
    # Reshape the reconstructed data into its original shape
    pc_reshaped = pc_transformed.reshape(n_bands, hs_5m_data.shape[1], hs_5m_data.shape[2])
    print(f"Pansharpened image shape is: {pc_reshaped.shape}")
    
    variance_ratios = pca.explained_variance_ratio_
    print(f"The sum of explained variance ratios is: {np.sum(variance_ratios)}")
    
    pc_comp = pca.components_
    print(f"The shape of the PCA component is: {pc_comp.shape}")
    
    # Return the reconstructed data
    return pc_reshaped, variance_ratios, pc_comp

# -------------------------------------------------

def pan_GS(pan_data, hs_5m_data):
    """
    Pansharpening function using the Gram-Schmidt Fusion algorithm.

    Args:
    - pan_data (numpy array): Panchromatic data (5m resolution)
    - hs_5m_data (numpy array): Upsampled hyperspectral data (5m resolution)

    Returns:
    - I_GS (numpy array): Pansharpened image using GS method
    """
    
    print("Using GS Pansharpening method")
    print(f"Pan data (5m resolution) shape is: {pan_data.shape}")
    print(f"Upsampled HS data (5m resolution) shape is: {hs_5m_data.shape}")
    print("---------------------")
    
    # Move the channels axis to the last dimension for hyperspectral data
    hs_5m_data  = np.moveaxis(hs_5m_data, 0, -1)
    
    # Compute mean for each band and subtract from input hyperspectral data
    means = np.mean(hs_5m_data, axis=(0, 1))
    image_lr = hs_5m_data-means
    
    # Compute syntetic intesity (I) and subtract mean
    I = np.mean(hs_5m_data, axis=2, keepdims=True)
    I0 = I-np.mean(I)
    
    # Move the channels axis to the last dimension for panchromatic data
    pan = np.moveaxis(pan_data, 0, -1)
    
    # Get the dimensions of input data
    M, N, c = pan.shape
    m, n, C = hs_5m_data.shape
    
    print(f"Pan image c: {c}, M: {M}, N: {N}")
    print(f"Upsampled HS image C: {C}, M: {m}, N: {n}")
    
    # Histogram equalization
    image_hr = (pan-np.mean(pan))*(np.std(I0, ddof=1)/np.std(pan, ddof=1))+np.mean(I0)
    
    # Compute gains (coefficients)
    g = []
    g.append(1)

    for i in range(C):
        temp_h = image_lr[:, :, i]
        covs = np.cov(np.reshape(I0, (-1,)), np.reshape(temp_h, (-1,)), ddof=1)
        g.append(covs[0,1]/np.var(I0))
    g = np.array(g)
    
    # Compute detail extraction
    delta = image_hr-I0
    deltam = np.tile(delta, (1, 1, C+1))
    
    # Compute fusion
    V = np.concatenate((I0, image_lr), axis=-1)
    
    # Expand dimensions of g to match the shape of V_hat
    g = np.expand_dims(g, 0)
    g = np.expand_dims(g, 0)
    
    # Tile g to match the shape of deltam
    g = np.tile(g, (M, N, 1))
    
    # Compute V_hat by adding the detail extraction matrix deltam multiplied by g to V
    V_hat = V + g*deltam
    
    # Extract the pansharpened image by removing the first channel (I0)
    I_GS = V_hat[:, :, 1:]
    
    # Center the pansharpened image around its mean and add back the mean of the original image
    I_GS = I_GS - np.mean(I_GS, axis=(0, 1))+means
    I_GS = np.clip(I_GS, a_min=None, a_max=1)
    
    # Move the channel axis to the front
    I_GS = np.moveaxis(I_GS, -1, 0)
    
    print(f"Output pansharpened image shape: {I_GS.shape}")
    
    return I_GS

#-----------------------------------------

#GSA 

def estimation_alpha(pan, hs, mode='global'):
    if mode == 'global':
        IHC = np.reshape(pan, (-1, 1))
        ILRC = np.reshape(hs, (hs.shape[0]*hs.shape[1], hs.shape[2]))
        
        alpha = np.linalg.lstsq(ILRC, IHC)[0]
        
    elif mode == 'local':
        patch_size = 32
        all_alpha = []
        print(pan.shape)
        for i in range(0, hs.shape[0]-patch_size, patch_size):
            for j in range(0, hs.shape[1]-patch_size, patch_size):
                patch_pan = pan[i:i+patch_size, j:j+patch_size, :]
                patch_hs = hs[i:i+patch_size, j:j+patch_size, :]
                
                IHC = np.reshape(patch_pan, (-1, 1))
                ILRC = np.reshape(patch_hs, (-1, hs.shape[2]))
                
                local_alpha = np.linalg.lstsq(ILRC, IHC)[0]
                all_alpha.append(local_alpha)
                
        all_alpha = np.array(all_alpha)
        
        alpha = np.mean(all_alpha, axis=0, keepdims=False)
        
    return alpha

def pan_GSA(pan, hs_data, hs_5m_data, mode='global'):
    
    print("Using GS Pansharpening method")
    print(f"Pan data (5m resolution) shape is: {pan.shape}")
    print(f"HS data (30m resolution) shape is: {hs_data.shape}")
    print(f"Upsampled HS data (5m resolution) shape is: {hs_5m_data.shape}")
    print("---------------------")
    
    # Move the channels axis to the last dimension for pan data
    pan  = np.moveaxis(pan, 0, -1)
    
    # Move the channels axis to the last dimension for hyperspectral data
    hs_data  = np.moveaxis(hs_data, 0, -1)
    
    # Move the channels axis to the last dimension for hyperspectral data
    hs_5m_data  = np.moveaxis(hs_5m_data, 0, -1)
    
    # Compute mean for each band and subtract from input hyperspectral data
    means = np.mean(hs_5m_data, axis=(0, 1))
    image_lr = hs_5m_data-means
    
    M, N, c = pan.shape
    m, n, C = hs_data.shape
    o_m, o_n, o_c = hs_5m_data.shape
    
    
    #remove means from u_hs
    means = np.mean(hs_5m_data, axis=(0, 1))
    image_lr = hs_5m_data-means
    
    #remove means from hs
    image_lr_lp = hs_data-np.mean(hs_data, axis=(0,1))
    
    #sintetic intensity
    image_hr = pan-np.mean(pan)
    image_hr0 = cv2.resize(image_hr, (n, m), cv2.INTER_CUBIC)
    image_hr0 = np.expand_dims(image_hr0, -1)
    
    alpha = estimation_alpha(image_hr0, np.concatenate((image_lr_lp, np.ones((m, n, 1))), axis=-1), mode=mode)
    
    I = np.dot(np.concatenate((image_lr, np.ones((M, N, 1))), axis=-1), alpha)
    
    I0 = I-np.mean(I)
    
    #computing coefficients
    g = []
    g.append(1)
    
    for i in range(C):
        temp_h = image_lr[:, :, i]
        c = np.cov(np.reshape(I0, (-1,)), np.reshape(temp_h, (-1,)), ddof=1)
        g.append(c[0,1]/np.var(I0))
    g = np.array(g)
    
    #detail extraction
    delta = image_hr-I0
    deltam = np.tile(delta, (1, 1, C+1))
    
    #fusion
    V = np.concatenate((I0, image_lr), axis=-1)
    
    g = np.expand_dims(g, 0)
    g = np.expand_dims(g, 0)
    
    g = np.tile(g, (M, N, 1))
    
    V_hat = V + g*deltam
    
    I_GSA = V_hat[:, :, 1:]
    
    I_GSA = I_GSA - np.mean(I_GSA, axis=(0, 1)) + means
    I_GSA = np.clip(I_GSA, a_min=None, a_max=1)
    
    # Move the channel axis to the front
    I_GSA = np.moveaxis(I_GSA, -1, 0)

    
    return I_GSA


