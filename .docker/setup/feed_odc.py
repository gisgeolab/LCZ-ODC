import datacube

from import_prisma import import_prisma 
from import_classified_images import import_classified_images
from import_climami import import_climami
from import_sentinel2 import import_sentinel2
from import_training_samples import import_training_samples
from import_ucp import import_ucp

dc = datacube.Datacube(app = 'my_app', config = '/home/asi/datacube_conf/datacube.conf')

list_of_products = dc.list_products(with_pandas=False)

def is_product_registered(product_name):
  return any(product['name'] == product_name for product in list_of_products)

if is_product_registered('PRISMA_Full_Bands'):
  print('PRISMA_Full_Bands already imported, skipping')
else:
  import_prisma()

if is_product_registered('classified_images'):
  print('classified_images already imported, skipping')
else:
  import_classified_images()

if is_product_registered('climami'):
  print('Climami already imported, skipping')
else:
  import_climami()

if is_product_registered('S2_Full_Bands'):
  print('Sentinel 2 already imported, skipping')
else:
  import_sentinel2()

if is_product_registered('training_samples'):
  print('Training Samples already imported, skipping')
else:
  import_training_samples()

if is_product_registered('UCP'):
  print('UCP already imported, skipping')
else:
  import_ucp()
