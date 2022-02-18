import netCDF4 as nc
import numpy as np
from skimage.io import imread
from datetime import datetime
from mjd2date import date2mjd
import time
from osgeo import gdal, osr
import glob

###  INPUT

nc_name = '/home/lauranec/Documents/Kyagar_velocities_TSX_75D.nc' #nom du fichier netcdf a creer
folder = '/home/lauranec/NAS/charriel/KyagarGlacier/orb075D_vel-128-64px-intCorr-11-110d+/EPSG32643/' #dossier contenant les geotiff a mettre sous format netcdf
geotif_ref = '/home/lauranec/NAS/charriel/KyagarGlacier/orb075D_vel-128-64px-intCorr-11-110d+/EPSG32643/velocity_grid_x_m-day2012-11-20_vs_2013-08-22_geo.tif' #geotiff de reference
sensor_name = 'TSX_orb75D'  # PAZ_asc ou PAZ_desc

### MAIN

# Creation of netcdf file
ds = nc.Dataset(nc_name, 'w', format='NETCDF4')

geotif = gdal.Open(geotif_ref, gdal.GA_ReadOnly)

# Create Title, Author, history
ds.Conventions = 'CF-1.6'
ds.title = 'Cube of Ice Velocity'
ds.author = 'S. Leinss, L. Charrier'  # a modifier
ds.history = 'Created ' + time.ctime(time.time())
ds.source = 'Intensity cross-correlation for SAR images, Normalized cross correlation of the brightness orientation for optical images'
ds.references = 'Kyagar glacier: Round, V., Leinss, S., Huss, M., Haemmig, C., & Hajnsek, I. (2017). Surge dynamics and lake outbursts of Kyagar Glacier, Karakoram. The Cryosphere, 11(2), 723‑739. https://doi.org/10.5194/tc-11-723-2017'

#Get the the x and y (in the image)
ulx, xres, xskew, uly, yskew, yres  = geotif.GetGeoTransform() #top-left x, w-e pixel resolution, 0, top left y, 0, n-s pixel resolution
lrx = ulx + (geotif.RasterXSize * xres)
lry = uly + (geotif.RasterYSize * yres)

print(geotif.RasterXSize,geotif.RasterYSize)
#Create Dimensions
ds.createDimension('x', geotif.RasterXSize)
ds.createDimension('y', geotif.RasterYSize)
ds.createDimension('z', None) #temps peut grandir infinement
ds.createDimension('nchar',9)

#Create variables

mapping = ds.createVariable('mapping', 'S1')

x = ds.createVariable('x', 'f8', ('x',))
x.units = 'm'
x.axis = "X"
x.long_name = 'Cartesian x-coordinate'
x.grid_mapping = 'UTM_projection'

y = ds.createVariable('y', 'f8', ('y',))
y.units = 'm'
y.axis = "Y"
y.long_name = 'Cartesian y-coordinate'
y.grid_mapping = 'UTM_projection'

vx = ds.createVariable('vx', 'f4', ('z', 'y', 'x',), fill_value=np.nan)
vx.units = 'm/y'
vx.long_name = 'x component of velocity vector '
vx.grid_mapping = 'UTM_projection'

vy = ds.createVariable('vy', 'f4', ('z', 'y', 'x',), fill_value=np.nan)
vy.units = 'm/y'
vy.long_name = 'y component of velocity vector '
vy.grid_mapping = 'UTM_projection'

##optionel
vv = ds.createVariable('vv', 'f4', ('z', 'y', 'x',), fill_value=np.nan)  # module
vv.units = 'm/y'
vv.long_name = 'velocity magnitude '
vv.grid_mapping = 'UTM_projection'

date1 = ds.createVariable('date1', 'i4', ('z',))
date1.units = 'day (Modified Julian Date)'
date1.content = 'Date of the first acquisition'

date2 = ds.createVariable('date2', 'i4', ('z',))
date2.units = 'day (Modified Julian Date)'
date2.content = 'Date of the second acquisition'

sensor = ds.createVariable('sensor','S1',('z','nchar'))
sensor.long_name = 'sensor used to acquire the images'

quality = ds.createVariable('quality', 'f4', ('z','y','x'),fill_value=np.nan)
quality.long_name = 'quality of vx and vy'
##a modifier
quality.content = 'A combination (average) of four different factors: 1) NCC: the normalized cross correlation value of the peak. 2) grad_quality: This is a measure which analyzes the shape of the peak. 3) 1.0 / (avg_peakdist > 1): Analyzes how broad the peak is. 4) 1.0 / (splitspec_peak_dist > 1): The peak of the cross correlation is caluclated in two separates parts of the spectrum (low and high frequency components).'
quality.grid_mapping = 'UTM_projection'

#Assign Latitude and Longitude
x[:] = np.arange(ulx, lrx, xres)
y[:] = np.arange(uly, lry, yres)

#Get projection
prj=geotif.GetProjection()
srs=osr.SpatialReference(wkt=prj)

mapping.grid_mapping_name = 'universal_transverse_mercator'
# mapping.utm_zone_number = 43
mapping.spatial_ref = srs.ExportToWkt()

#Files the variables
files = glob.glob(
    f'{folder}velocity_grid_x_m-day*')  # cherche tous les fichiers qui ont une certaine chaine de caractere
z = 0
for vx_file in files:
    print(vx_file)
    # im = imread(vx_file)
    # print(im.shape)
    #
    end_file = vx_file.split('velocity_grid_x_m-day')[-1]
    begin_file = vx_file.split('velocity_grid_x')[0]
    vy_file = f'{begin_file}velocity_grid_y_m-day{end_file}'
    vv_file = f'{begin_file}velocity_grid_abs_m-day{end_file}'
    quality_file = f'{begin_file}velocity_grid_quality_{end_file}'
    geotif = gdal.Open(vx_file)

    # on rempli chaque variable avec un tableau de valeur, qui correspond a un z donne
    vx[z, :, :] = imread(vx_file)
    vy[z, :, :] = imread(vy_file)
    vv[z, :, :] = imread(vv_file)
    quality[z, :, :] = imread(quality_file)
    # convert date to julian date, keep only the second returned number
    date1[z] = date2mjd(datetime.strptime(vx_file.split('/')[-1].split('day')[-1][:10], '%Y-%m-%d').date())
    date2[z] = date2mjd(datetime.strptime(vx_file.split('/')[-1].split('day')[-1][14:24], '%Y-%m-%d').date())
    sensor[z] = nc.stringtochar(np.array(sensor_name, 'S9'))
    z += 1

ds.nx = geotif.RasterXSize  # nombre de colonne
ds.ny = geotif.RasterYSize  # nombre de ligne
ds.nz = z  # nombre de couche
ds.i = ulx  # top-left pixel (optionnel) colonne
ds.j = uly  # top-left pixel (optionnel) ligne
ds.proj4 = srs.GetAttrValue('GEOGCS')

ds.close()  # ferme le fichier

# Merge

# import xarray as xr
#
# L8_147 = xr.open_dataset('/home/lauranec/Documents/Donnees_Sylvan/Kyagar-velocity/concate_L8.nc')
# L8_148 = xr.open_dataset('/home/lauranec/Documents/Donnees_Sylvan/Kyagar-velocity/Kyagar_velocities_S1_asc129.nc')
# xr.merge([L8_147, L8_148])
#
# ds = xr.open_mfdataset('/home/lauranec/Documents/Donnees_Sylvan/Kyagar-velocity/*L8*.nc', combine='nested',
#                        concat_dim='z')
# ds.to_netcdf('/home/lauranec/Documents/Donnees_Sylvan/Kyagar-velocity/concate_L8.nc')
