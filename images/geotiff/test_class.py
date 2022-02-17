#%matplotlib qt
from class_geotiff import geotiff
from class_stats import Statistical_Xarea

############## DESCENDANT ###############

desc=geotiff('../../DATASET/Sentinel-1/*DES*.tif',
             reproject="EPSG:4326",resample_factor=1/4,maxf=2)
desc.read_dataset()

date_des = desc.geotiffs_ds.time[0]
stats = Statistical_Xarea(desc,date_des,"VV","DESCENDANT ")
stats = Statistical_Xarea(desc,date_des,"VV","DESCENDANT ",[(6.869782,44.646926),(6.877764,44.651872)])

desc.plot_timeseries(4,namef="test",colormap="gray")
desc.plot3D(date_des)
desc.plot2D(date_des)


############## ASCENDANT ###############

ascd = geotiff('../../DATASET/Sentinel-1/*AS*.tif',
               reproject="EPSG:4326",resample_factor=1/4,maxf=-1)
ascd.read_dataset()

date_asc = ascd.geotiffs_ds.time[0]
#stats = Statistical_Xarea(ascd,date_asc,"VV","ASCENDANT ")
stats = Statistical_Xarea(ascd,date_asc,"VV","ASCENDANT ",[(6.869782,44.646926),(6.877764,44.651872)])

ascd.plot_timeseries(4,namef="test",colormap="gray")
ascd.plot3D(date_asc)
ascd.plot2D(date_asc)



############ RASTER GOOGLE EARTH PRO
desc.export_raster(date_des,"VV",None,"desc")
ascd.export_raster(date_asc,"VV",None,"ascd")
########################################

F=geotiff(f'../../DATASET/IFREMER/GeoTIFF/{a}/*.tiff',reproject=None,resample_factor=None,maxf=-1,band_name={1:'VV'})
F.extract_data()# add selection
F.histogram_XL('test')