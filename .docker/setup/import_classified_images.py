import os
from glob import iglob
import rasterio
from os.path import join

import warnings
warnings.filterwarnings('ignore')

from termcolor import colored

def import_classified_images():
  print('Importing classified images...')
  
  file_basepath = '/home/asi/data/classified_images'
  list_files = sorted(list(iglob(join(file_basepath, '*.tif*'), recursive=True)))

  file_names = []
  for path in list_files:
      file_name = path.split('/')[-1]
      file_name = file_name.replace('_RandomForest', '')
      file_name = file_name.replace('_medianfilter_30m.tif', '')
      file_names.append(file_name)

  file_dict = {}
  for key in file_names:
      for value in list_files:
          file_dict[key] = value
          list_files.remove(value)
          break

  def get_key_from_value(dictionary, target_value):
    matching_keys = [key for key, value in dictionary.items() if value == target_value]
    return matching_keys[0] if matching_keys else None

  key = get_key_from_value(file_dict, '/home/asi/data/classified_images/classified_RandomForest_20230209_medianfilter_30m.tif')

  list_files = sorted(list(iglob(join(file_basepath, '*.tif*'), recursive=True)))

  p_name = "classified_images"
  text_p0 = """name: """ + p_name + """
description: Classified images using scikit-learn Random Forest Classified
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

  text_p2_1 = """
    - name: classified_images
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
          
  print(colored('All classified images products have been updated into the ODC','green'))

  for file in list_files:
      date_uuid = file[file.find('202'):file.find('202')+8]
      print(date_uuid)
      src = rasterio.open(file)
      xmin,ymin,xmax,ymax = src.bounds
      affine = src.transform
      shape = src.shape
      text_p1 = ("""$schema: https://schemas.opendatacube.org/dataset

id: 60000000-0000-0000-0000-0000"""+ str(date_uuid) +"""

product:
  name: classified_images
  href: None
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

      text_p2_1 = """
  classified_images:
      path: """+str(file)+""""""
      text_p2 = text_p2 + text_p2_1
      
      text_p3 = """

properties:
  odc:file_format: GeoTIFF
  datetime: """+str(date_uuid)+"""T10:30:00Z
    """
      yaml_text = open(file.replace('.tif', '.yaml'), "wt")
      yaml_text.write(text_p1+text_p2+text_p3)
      yaml_text.close()
      os.system("""cd /home/asi/datacube_conf; 
      datacube dataset add """+file.replace('.tif', '.yaml'))
      
  print(colored('All classified image products have been imported into the ODC','green'))
