from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import ndimage
import numpy as np
import rasterio
import tqdm

def open_raster(name):
    seai_dtst=[]
    size_data=[]
    for i in tqdm.tqdm(name):
        with rasterio.open(i) as x:
            size_data.append([x.width,x.height])
            seai_dtst.append(x.read()[:,:,:])
    return np.array(size_data),seai_dtst

def adjust_size_raster(size,dataset):
    minW = size[:,0].min()
    minH = size[:,1].min()
    resize=[]
    for array in tqdm.tqdm(dataset):
        resize.append(np.moveaxis(array[:minH,:minW],0,2))
    return np.array(resize)

def compute_eight_dimensional_feature(image):
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x,y)
    Ix = ndimage.sobel(image,axis=1,mode='constant')
    Ixx = ndimage.sobel(Ix,axis=1,mode='constant')
    Iy = ndimage.sobel(image,axis=0,mode='constant')
    Iyy = ndimage.sobel(Iy,axis=0,mode='constant')
    I_abs = np.hypot(np.abs(Ix), np.abs(Iy))
    A = np.arctan2(np.abs(Iy), np.abs(Ix))

    return np.dstack([X, Y, np.abs(Ix), np.abs(Iy),
                        I_abs, np.abs(Ixx), np.abs(Iyy), A])

def DisplayScatteringVector_RGB(S2_Vectorized,Dyn=4,display_var=True):
    im0 = S2_Vectorized
    im = im0 - np.min(im0[:,:,:])

    imp1 = np.abs(im[:,:,0])
    imp2 = np.abs(im[:,:,1])
    imp3 = np.abs(im[:,:,2])

    D1 = Dyn*imp1.std()
    D2 = Dyn*imp2.std()
    D3 = Dyn*imp3.std()

    imp1[imp1>D1] = D1
    imp2[imp2>D2] = D2
    imp3[imp3>D3] = D3


    imp1 = imp1/D1
    imp2 = imp2/D2
    imp3 = imp3/D3

    im[:,:,0] = imp1
    im[:,:,1] = imp2
    im[:,:,2] = imp3

    imk3rgb = np.abs(im)
    if display_var:
        fig,ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].imshow(imk3rgb)
        for channel, color in enumerate(['red', 'green', 'blue']):
            ax[1].hist(im0[..., channel].ravel(),1000, alpha=0.5,facecolor=color, label='%s channel'%color,)
        plt.legend()
        plt.savefig('display_SAR.svg')
    return imk3rgb


def animation_image(data_adj,downs=(1,1,1,1),save=False):
    x=[]
    for i in tqdm(range(data_adj.shape[0])):
        x.append(DisplayScatteringVector_RGB(data_adj[i],Dyn=4,display_var=False))
    x=np.array(x)
    x2=block_reduce(x, block_size=downs, func=np.mean)
    if save:
        fig = px.imshow(x2, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
        fig.write_html("animation.html")
    return x2

def DisplayScatteringXarray_band(band,Dyn=4):
    im = band - band.min(axis=(1,2))
    imp1 = np.abs(im)
    D1 = Dyn*imp1.std(axis=(1,2))
    imp1=imp1.where(imp1<=D1,D1)
    imp1 = imp1/D1
    imk3rgb = np.abs(imp1)
    return imk3rgb