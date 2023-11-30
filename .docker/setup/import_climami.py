import os
from glob import iglob
import rasterio
from os.path import join

import warnings
warnings.filterwarnings('ignore')

from termcolor import colored

def import_climami():
  print('Importing Climami...')

  file_basepath = '/home/asi/data/meteo/climami'
  list_files = sorted(list(iglob(join(file_basepath, '*.tif*'), recursive=True)))

  file_names = []
  for path in list_files:
      file_name = path.split('/')[-1]
      file_name = file_name.replace('.tif', '')
      file_names.append(file_name)

  file_dict = {}
  for key in file_names:
      for value in list_files:
          file_dict[key] = value
          list_files.remove(value)
          break
  file_dict

  list_files = sorted(list(iglob(join(file_basepath, '*.tif*'), recursive=True)))

  p_name = "climami"
  text_p0 = """name: """ + p_name + """
description: raster data retrived from CLIMAMI Project (Osservatorio Meteorologico Milano).
metadata_type: eo3

metadata:
    product:
        name: """ + p_name + """
        format: GeoTIFF

storage:
    crs: EPSG:32632
    dimension_order: [time, x, y]
    resolution: 
        x: 30 
        y: -30"""

  file = list_files[0]
  date_uuid = file[file.find('202'):file.find('202')+8]
  src = rasterio.open(file) 
  xmin,ymin,xmax,ymax = src.bounds
  affine = src.transform
  shape = src.shape
  text_p2_1 = """"""
  text_p2 = """"""
  text_p1 = ("""

measurements:""")
  """"""
  for band in file_names:
      text_p2_1 = """
    - name: """+band+"""
      dtype: float32
      nodata: NaN
      units: ''
  """
      text_p2 = text_p2 + text_p2_1
  text_p3 = text_p0 + text_p1 + text_p2
  yaml_text = open('/home/asi/datacube_conf/'+p_name+'.yaml', "wt")
  yaml_text.write(text_p3)
  yaml_text.close()
  os.system("""cd /home/asi/datacube_conf; 
  datacube product add /home/asi/datacube_conf/"""+p_name+""".yaml""")
          
  print(colored('All Climami products have been updated into the ODC','green'))

  for file in list_files:
      filename='/home/asi/data/meteo/climami/climami.yaml'
      date_uuid = '20230209'
      print(file)
      src = rasterio.open(file)
      xmin,ymin,xmax,ymax = src.bounds
      affine = src.transform
      shape = src.shape
      text_p1 = ("""$schema: https://schemas.opendatacube.org/dataset

id: 50000000-0000-0000-0000-0000"""+ str(date_uuid) +"""

product:
  name: climami
  href: https://www.progettoclimami.it/
  format: GeoTIFF

crs: EPSG:32632

geometry:
  type: Polygon
  coordinates: [[["""+str(xmin)+""", """+str(ymin)+"""], ["""+str(xmin)+""", """+str(ymax)+"""], ["""+str(xmax)+""", """+str(ymax)+"""], ["""+str(xmax)+""", """+str(ymin)+"""], ["""+str(xmin)+""", """+str(ymin)+"""]]]

grids:
  default:
    shape: ["""+str(shape[0])+""","""+str(shape[1])+"""]
    transform: ["""+str(affine[0])+""","""+str(affine[1])+""","""+str(affine[2])+""","""+str(affine[3])+""","""+str(affine[4])+""","""+str(affine[5])+""", 0,0,1]

lineage: {}
measurements:
""")
      text_p2 = """"""
      for band in file_names:
          text_p2_1 = """
      """+str(band)+""":
          path: """+str(file_dict[band])+""""""
          text_p2 = text_p2 + text_p2_1
      
      text_p3 = """
properties:
    odc:file_format: GeoTIFF
    datetime: """+str(date_uuid)+"""T10:30:00Z
      """
      yaml_text = open(filename.replace('.tif', '.yaml'), "wt")
      yaml_text.write(text_p1+text_p2+text_p3)
      yaml_text.close()
      os.system("""cd /home/asi/datacube_conf; 
      datacube dataset add """+filename.replace('.tif', '.yaml')) 
          
  print(colored('All Climami products have been imported into the ODC','green'))
