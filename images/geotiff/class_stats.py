from matplotlib.widgets import RectangleSelector
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

class Statistical_Xarea:
    def __init__(self, data, date,bande,name="",selection=[]):
        fig, ax = plt.subplots(3,2)
        plt.style.use('bmh')
        fig.suptitle('Visualisation spatio-temporelle stats zones',fontsize=12,fontweight='bold')
        self.fig = fig
        self.axes = ax
        self.data = data.geotiffs_dsorg[bande]
        self.img = data.geotiffs_ds[bande]
        self.date = date
        self.ind = 25
        self.selection = selection
  
        gs = ax[0, 0].get_gridspec()
        # remove the underlying axes
        for ax in self.axes[2, :2]:
            ax.remove()
        
        self.axbig = fig.add_subplot(gs[2, :2])

        for ax in self.axes[:2, 0]:
            ax.remove()
        self.ax_imsh = fig.add_subplot(gs[:2, 0])
        self.canvas = self.ax_imsh.figure.canvas#ax[0,0]

        self.cb=None

        max_date = self.img.sel(time=date).max().values
        imsh = self.ax_imsh.pcolormesh(self.img.sel(time=date).x.values,self.img.sel(time=date).y.values,self.img.sel(time=date).values,norm=LogNorm(vmax=max_date),shading='auto',rasterized=True,snap=True)
        self.ax_imsh.set_title(f"{name}{bande}\n{date.values}",fontdict={'fontsize': 10})
        
        self.fig.colorbar(imsh, ax=self.ax_imsh, label="# points",aspect=10)

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    def update_hist(self,i,j):
        self.axes[i,j].clear()
        self.axes[i,j].hist(np.ravel(self.time_data),log=True,bins=self.ind,label=f"space_stats bins:{self.ind}",density=False,alpha=0.5)
        self.axes[i,j].hist(np.ravel(self.space_data),log=True,bins=self.ind,label=f"temporal_stats bins:{self.ind}",density=False,alpha=0.5)
        self.axes[i,j].legend()
        self.axes[i,j].figure.canvas.draw_idle()
        self.axes[i,j].set_title("Histogram1D",fontdict={'fontsize': 12,'fontweight':'bold'})

    def update_2Dhist(self):
        self.axbig.clear()
        if self.cb != None:
            self.cb.remove()
        data=self.select_data.stack(z=("x", "y"))
        data = data.assign_coords(z=np.arange(data.z.size))
        amin = data.min().values
        amax = data.max().values
        H=np.zeros((data.time.size,self.ind))
        for idx, val in enumerate(data):
            H[idx,:],yedge= np.histogram(val.values,range=(amin,amax),bins=self.ind)

        pcm = self.axbig.pcolormesh(data.time.values,yedge[:-1],H.T,norm=LogNorm(vmax=np.max(H)),shading='nearest',rasterized=True)
        self.cb = self.fig.colorbar(pcm, ax=self.axbig, label="# points",aspect=100, location='bottom',orientation="horizontal")
        self.axbig.set_title(f"2D_histogram bins:{self.ind}",fontdict={'fontsize': 12,'fontweight':'bold'})

    def on_scroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1)
        else:
            if self.ind>5:
                self.ind = (self.ind - 1)

        self.update_hist(1,1)
        self.update_2Dhist()

    def callback(self, eclick,erelease):
        if len(self.selection)>0:
            self.x1, self.y1 = self.selection[0]
            self.x2, self.y2 = self.selection[1]
        else:
            self.x1, self.y1 = eclick.xdata, eclick.ydata
            self.x2, self.y2 = erelease.xdata, erelease.ydata
        self.ax_imsh.patches=[]
        self.ax_imsh.add_patch(patches.Rectangle((self.x1, self.y1), self.x2-self.x1, self.y2-self.y1,fill=True,alpha=0.5,color="red"))
        if (self.data.y[0]-self.data.y[-1])<0:
            select_data = self.data.sel(x=slice(self.x1,self.x2),y=slice(self.y1,self.y2))
        else:
            select_data = self.data.sel(x=slice(self.x1,self.x2),y=slice(self.y2,self.y1))
        self.select_data=select_data
        self.space_data = select_data.sel(time=self.date)
        self.time_data = select_data.mean('time')
        self.axes[0,1].imshow(self.space_data,aspect='equal')
        
        self.update_hist(1,1)
        self.update_2Dhist()

        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso

    def on_press(self, event):
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        
        self.lasso = RectangleSelector(event.inaxes,self.callback,drawtype='box',
                                        button=[1, 3],  # don't use middle button
                                        minspanx=5, minspany=5,
                                        spancoords='pixels')
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)