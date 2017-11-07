import pickle
import numpy as np
from sklearn.manifold import TSNE
from bokeh.models import (LassoSelectTool, PanTool,
                          ResetTool,
                          HoverTool, WheelZoomTool, ColumnDataSource)
TOOLS = [LassoSelectTool, PanTool, WheelZoomTool, ResetTool]
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral6
import os
import pandas as pd
import base64

'''
dirs       = Array of directorys for classes to plot
nPerClass  = Number of images to be used from each class (2 classes)
activFiles = Array of pickle files containing network activations
'''

'''
Genres defined as:
jazz  = 0
dance = 1
rock  = 2
rap   = 3
metal = 4
'''

tooltip = """
        <div>
            <div>
                <img
                src="@image_files" height="128"  width="128" alt="image"
                style="float: left; margin: 0px 15px 15px 0px; image-rendering: pixelated;"
                border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px;">@source_filenames</span>
            </div>
        </div>
           """

class tsneBokeh:

    def __init__(self, dirs, nPerClass, activFiles, labels=["0", "1"]):
        self.n = nPerClass
        self.dirs = dirs
        self.tooltip = tooltip
        self.activFiles = activFiles
        self.labels = labels
        self.dics = []
        self.activs = []
        self.URIs = []
        self.genres = []
        self.nClass = len(labels)
        self.tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
        output_file("tsnebokeh.html")
        self.colors = Spectral6
        self.getActivations()
        self.getURIs()

    def getActivations(self):
        try:
            for p in self.activFiles:
                self.dics.append(pickle.load(open(p, "rb")))
            for d in self.dics:
                activs = d["activs"][0:self.n]
                genres = (d["genres"][0:self.n])
                for a in activs:
                    self.activs.append(a)
                for g in genres:
                    self.genres.append(g)
        except Exception as e:
            print(e)

    def imgToData(self, p):
        data = base64.b64encode(open(p, 'rb').read()).decode('utf-8').replace('\n', '')
        uri = 'data:image/png;base64,{0}'.format(data)
        return uri

    def getURIs(self):
        for d in self.dirs:
            uri = []
            files = os.listdir(d)[0:self.n]
            for f in files:
                enc = self.imgToData(os.path.join(d,f))
                uri.append(enc)
            self.URIs.append(uri)

    def createPlot(self):
        res = self.tsne.fit_transform(self.activs)
        hover = HoverTool(tooltips=self.tooltip)
        tools = [t() for t in TOOLS] + [hover]
        p=figure(plot_width=1200, plot_height=800, title=None,
                 toolbar_location="below", tools=tools)

        # Count through each genre for both real and generated to find the
        # right data points to plot in different colours
        for j in range(0, 2):
            for i in range(0, int(self.nClass/2)):
                df = pd.DataFrame()
                # If generated need first half of the list of activation values
                if j == 0:
                    gen = self.genres[0:self.n]
                    ind = [ix for ix, value in enumerate(gen) if value == i]
                    df["x-tsne"] = res[0:self.n, 0][ind]
                    df["y-tsne"] = res[0:self.n, 1][ind]
                # If real need second half of the list of activation values
                elif j == 1:
                    gen = self.genres[self.n:]
                    ind = [ix for ix, value in enumerate(gen) if value == i]
                    df["x-tsne"] = res[self.n:, 0][ind]
                    df["y-tsne"] = res[self.n:, 1][ind]
                df["image_files"] = [self.URIs[j][k] for k in ind]
                df['color'] = self.colors[i]
                df['label'] = self.labels[i*(1+j)]
                if j == 0:
                    p.cross('x-tsne', 'y-tsne', size=8, color='color',
                             legend='label', source=ColumnDataSource(df))
                elif j == 1:
                    p.circle('x-tsne', 'y-tsne', size=8, color='color',
                             legend='label', source=ColumnDataSource(df))
        show(p)


if __name__ == '__main__':
    files = ["gen_activ2.p", "real_activ.p"]
    dirs = ["./generated", "./real"]
    labels = ["Generated Jazz", "Generated Dance", "Generated Rock",  \
              "Generated Rap", "Generated Metal", "Real Jazz", "Real Dance", \
              "Real Rock", "Real Rap", "Real Metal"]
    tsb = tsneBokeh(dirs, 1000, files, labels)
    tsb.createPlot()
