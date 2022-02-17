from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import dask.array as da
import xarray as xr
import numpy as np
import rioxarray
import tqdm
import glob

from SAR_raster_open_func import DisplayScatteringXarray_band
from rasterio.enums import Resampling


def resample_grid(x,y,dec=10):
    yg=np.array([[y[i,j]for i in range(0,(y.shape[0]-1),dec)]for j in range(0,(y.shape[1]-1),dec)])
    xg=np.array([[x[i,j]for i in range(0,(x.shape[0]-1),dec)]for j in range(0,(x.shape[1]-1),dec)])
    return xg.ravel(),yg.ravel()

def downs_reproj(xds,up_down_factor,projection="EPSG:4326"):
    new_width = int(xds.rio.width * up_down_factor)
    new_height = int(xds.rio.height * up_down_factor)
    return xds.rio.reproject(
        projection,
        shape=(new_height, new_width),
        resampling=Resampling.bilinear,
    )

def resampling_array(xds,up_down_factor):
    new_width = int(xds.rio.width * up_down_factor)
    new_height = int(xds.rio.height * up_down_factor)
    return xds.rio.reproject(
        xds.rio.crs,
        shape=(new_height, new_width),
        resampling=Resampling.bilinear,
    )

def reproject_array(xds,projection):
    return xds.rio.reproject(projection)

def plot_3Dmap(geotiffs_da,bands,date_t,max_z,dec_grid,name_file,db):
    x=geotiffs_da.sel(time=date_t)[bands].x
    y=geotiffs_da.sel(time=date_t)[bands].y
    if db:
        z=10*np.log(geotiffs_da.sel(time=date_t)[bands].values)
    else:
        z=geotiffs_da.sel(time=date_t)[bands].values
    x2,y2=np.meshgrid(x,y)

    csmap = [[0.,     'rgb(0,0,0)'], 
            [1./500, 'rgb(25,25,25)'],             
            [1./100, 'rgb(50,50,50)'],
            [1./25,  'rgb(100,100,100)'],
            [0.15,   'rgb(150,150,150)'],
            [0.5,    'rgb(200,200,200)'],
            [1.,     'rgb(250,250,250)']]
    mapbox_token = 'pk.eyJ1IjoibWF0dHVzbWIiLCJhIjoiY2t2aHlnMXBzMDdhbjJ1dGszOGo0OXc5cCJ9.FZlBUHiVzBsZVX27lsfU-g'
    fig = make_subplots(rows=1, cols=2,specs=[[{"type": "mapbox"}, {"type": "surface"}]],subplot_titles=(f"Carte de la zone étudiée + pixels > {max_z} ", "Carte 3D zone"))
    fig.add_trace(go.Scattermapbox(mode = "markers+lines",lon = [x.max(), x.min(), x.min(), x.max(),x.max()],
                                                          lat = [y.max(), y.max(),y.min(), y.min(),y.max()],
                                                          marker = { 'size': 10, 'color': "orange" },name="Zone"), 1, 1
                )
    fig.add_trace(go.Scattermapbox(lat=y2[z>max_z], lon=x2[z>max_z],hovertext=z[z>max_z],
                                marker={'size': 10,'color':(z[z>max_z]),"colorscale":"Spectral",
                                        "reversescale":True,"colorbar":dict(len=0.8,x=-0.1)},name='Pmax'), 1, 1
                )
    xg,yg =resample_grid(x2,y2,dec_grid)
    fig.add_trace(go.Scattermapbox(mode = "markers",lat=yg, lon=xg,
                                marker={'size': 5,'color': "green","opacity":0.0},name='Grid'), 1, 1
                )
    fig.add_trace(go.Surface(z=z,x=x,y=y,colorscale=csmap,
                            cmin=0,cauto=False,cmax=z.mean()+3*z.std()), 1, 2
                )
    fig.update_layout(mapbox = {'accesstoken':mapbox_token,
                                'center': {'lon':float(x.mean())  , 'lat':float(y.mean()) },
                                'zoom': 10,
                                'style':"outdoors"},
                    scene = dict( xaxis = dict(showbackground=False, ticks='outside',tickwidth=2,),
                                    yaxis = dict(showbackground=False,ticks='outside',tickwidth=2,),
                                    zaxis = dict(ticks='outside',tickwidth=2),
                                    yaxis_title='Latitude',
                                    zaxis_title='pixel value',
                                    xaxis_title='Longitude',
                                    camera_eye=dict(x=0.1, y=-0.25, z=2),
                                ),
                    showlegend = True,
                    legend = dict(x=-0.12,y=1.1,yanchor="top"),
                    title_text=f"Résumé du {date_t.values} en bande {bands}"
                    )
    fig.write_html(f'{name_file}.html')


class geotiff:
    def __init__(self, path, reproject=None, resample_factor=None,
                       band_name={1: 'VV',2: 'VH',3: 'VH_VV'}, keep_org=False, save=True,maxf=-1):
        self.path = path
        self.reproject = reproject
        self.resample_factor = resample_factor
        self.band_name = band_name
        self.keep_org = keep_org
        self.save = save
        self.maxf = maxf
        plt.style.use('bmh')

    def _extract_time(self):
        format = "%Y:%m:%d %H:%M:%S"
        self.list_file = glob.glob(self.path)[:self.maxf]
        try:
            date_temp =[xr.open_rasterio(i).attrs['TIFFTAG_DATETIME'] for i in self.list_file]
            dt_object = np.array([datetime.datetime.strptime(i, format) for i in date_temp],dtype=np.datetime64)
        except:
            dt_object=np.arange(len(self.list_file))
        self.time_var = xr.Variable('time',dt_object)

    def read_xarray(self):
        self._extract_time()
        self.geotiffs_org = xr.concat([xr.open_rasterio(i) for i in tqdm.tqdm(self.list_file)],dim=self.time_var)
        
    def processing(self,*argv):
        if argv:
            self.geotiffs_org = argv

        if (self.reproject==None)and(self.resample_factor!=None):
            self.geotiffs_da = xr.concat([resampling_array(i,self.resample_factor)for i in tqdm.tqdm(self.geotiffs_org)],dim=self.time_var)
        elif (self.reproject!=None)and(self.resample_factor==None):
            self.geotiffs_da = xr.concat([reproject_array(i,self.reproject)for i in tqdm.tqdm(self.geotiffs_org)],dim=self.time_var)
            self.geotiffs_org = self.geotiffs_da
        elif (self.reproject!=None)and (self.resample_factor!=None):
            self.geotiffs_da = xr.concat([downs_reproj(i,self.resample_factor,self.reproject)for i in tqdm.tqdm(self.geotiffs_org)],dim=self.time_var)
            self.geotiffs_org = xr.concat([reproject_array(i,self.reproject)for i in tqdm.tqdm(self.geotiffs_org)],dim=self.time_var)
        else:
            self.geotiffs_da = self.geotiffs_org
        self.geotiffs_da = self.geotiffs_da.sortby("time")

    def _array2dataset(self):
        self.geotiffs_dsorg = self.geotiffs_org.to_dataset('band')
        self.geotiffs_dsorg = self.geotiffs_dsorg.rename(self.band_name)
        self.geotiffs_ds = self.geotiffs_da.to_dataset('band')
        self.geotiffs_ds = self.geotiffs_ds.rename(self.band_name)

    def read_dataset(self):
        self.read_xarray()
        self.processing()
        self._array2dataset()

    def plot2D(self,date,band=None,comap="seismic",scale=3):
        if band==None:
            imag = self.geotiffs_da.sel(time=date).sum(axis=0)
            imag.plot.pcolormesh(cmap=comap,vmax=imag.mean()+scale*imag.std())
        else:
            imag = self.geotiffs_da.sel(time=date)[band]
            imag.plot.pcolormesh(cmap=comap,vmax=imag.mean()+scale*imag.std())

    def plot_timeseries(self,dynamic,colormap="gray",namef=""):
        for n_band in self.geotiffs_ds.data_vars:
            self.geotiffs_ds[n_band] = DisplayScatteringXarray_band(self.geotiffs_ds[n_band],dynamic)
        if self.save:                
            for n_band in self.geotiffs_ds.data_vars:
                fig = px.imshow(self.geotiffs_ds[n_band], animation_frame='time',
                                color_continuous_scale=colormap,aspect="equal",
                                labels={"y":"latitude","x":"longitude"},origin="lower",
                                binary_compression_level=9,binary_format='jpg')
                fig.write_html(f"{n_band}_{namef}.html")

    def plot3D(self,date_t,bands="VV",max_value=5,dec_grid=5,namef="3D",db=False):
        plot_3Dmap(self.geotiffs_ds,bands,date_t,max_value,dec_grid,namef,db)

    def export_raster(self,date_t,bands="VV",dynamic=None,name=""):
        if dynamic==None:
            log_tif = self.geotiffs_dsorg[bands]
        else:
            log_tif = DisplayScatteringXarray_band(self.geotiffs_dsorg[bands],dynamic)
            log_tif=np.log10(log_tif.sel(time=date_t)+1)
        log_tif = (log_tif-log_tif.min())/(log_tif.max()-log_tif.min())*65535
        log_tif.rio.to_raster(f"raster_earth_repro_{name}{date_t.values}.tif",dtype=np.uint16)

    def extract_data(self):
        self._extract_time()
        self.geotiffs_org=[da.from_array(xr.open_rasterio(i).values.ravel()) for i in tqdm.tqdm(self.list_file)]
        self.geotiffs_org=da.concatenate(self.geotiffs_org)
    
    def histogram_XL(self,lab):
        self.extract_data()
        fig,ax2 = plt.subplots(1)
        ma=self.geotiffs_org.max().compute()
        mi=self.geotiffs_org.min().compute()
        h, bins = da.histogram(self.geotiffs_org,bins=500,range=[mi,ma],density=True)       
        ax2.bar(bins[:-1], h.compute(),width=np.diff(bins), align="edge",alpha=0.5,label=lab)
        ax2.legend()
        plt.show()