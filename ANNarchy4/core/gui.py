try:
    import Tkinter as Tk
except:
    print 'Error: Tkinter is not installed.'
    raise ImportError
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
    from matplotlib.figure import Figure
    import matplotlib.gridspec as gridspec
except:
    print 'Error: matplotlib is not installed.'
    raise ImportError

import numpy as np
import threading

class PlotThread(threading.Thread):

    def __init__(self, plotData, persist):
        threading.Thread.__init__(self)
        self.plotData = plotData
        self.daemon=(not persist)
        self.start()

    def run(self):
        self.root = Tk.Tk()
        self.root.wm_title("ANNarchy 3.1")
        
        # Initialize the figure
        self.fig=Figure()
        self.h=self.create_figure()

        # a tk.DrawingArea
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        
        # Infinite loop
        Tk.mainloop()
        
    def create_matrix(self, id):       
        pd = getattr(self.plotData[id]['pop'].cyInstance, self.plotData[id]['var'])

        geo = self.plotData[id]['pop'].geometry
	return pd.reshape((geo[0], geo[1]))
        
    def create_figure(self):
	h = []

	nX = 1			# TODO: determine good sizes, or maybe user defined?
	nY = len(self.plotData) 
        gs = gridspec.GridSpec(nX, nY)

	for p in range(nY):
            ax=self.fig.add_subplot(gs[0,p])
            
            d = ax.imshow(self.create_matrix(p), interpolation='nearest', cmap=matplotlib.cm.hot, vmin=0.0, vmax=1.0)
            if 'name' in self.plotData[p]:
               ax.set_title(self.plotData[p]['name'])
            else:
               ax.set_title('plot_'+str(p))
            h.append(d)

        self.root.after(50, self.update_figure)
        return h
        
    def update_figure(self):
        #
        #update all current handles
        for id in range(len(self.h)):
            m = self.create_matrix(id)
            self.h[id].set_data(m) 
        
        self.canvas.draw()
        self.root.after(50, self.update_figure)  
