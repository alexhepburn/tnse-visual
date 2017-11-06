import pickle
import numpy as np
from sklearn.manifold import TSNE
from bokeh.models import (LassoSelectTool, PanTool,
                          ResetTool,
                          HoverTool, WheelZoomTool, ColumnDataSource)
TOOLS = [LassoSelectTool, PanTool, WheelZoomTool, ResetTool]
from bokeh.plotting import figure, output_file, show
import os
import pandas as pd
import base64

'''
dirs       = Array of directorys for classes to plot
nPerClass  = Number of images to be used from each class (2 classes)
activFiles = Array of pickle files containing network activations
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
        self.nClass = len(dirs)
        self.tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
        output_file("tsnebokeh.html")
        self.colors = ['navy', 'firebrick']
        self.getActivations()
        self.getURIs()

    def getActivations(self):
        try:
            for p in self.activFiles:
                self.dics.append(pickle.load(open(p, "rb")))
            for d in self.dics:
                activs = d["activs"][0:self.n]
                for a in activs:
                    self.activs.append(a)
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
        activations = []
        dataFrames = []
        sources = []
        res = self.tsne.fit_transform(self.activs)
        for i in range(0, self.nClass):
            df = pd.DataFrame()
            if i == 0:
                df["x-tsne"] = res[0:self.n, 0]
                df["y-tsne"] = res[0:self.n, 1]
            elif i == 1:
                df["x-tsne"] = res[self.n:, 0]
                df["y-tsne"] = res[self.n:, 1]
            print(len(self.URIs))
            df["image_files"] = self.URIs[i]
            df['color'] = self.colors[i]
            df['label'] = self.labels[i]
            sources.append(ColumnDataSource(df))
            dataFrames.append(df)
        hover = HoverTool(tooltips=self.tooltip)
        tools = [t() for t in TOOLS] + [hover]
        p=figure(plot_width=800, plot_height=800, title=None, toolbar_location="below", tools=tools)
        for s in sources:
            p.circle('x-tsne', 'y-tsne', size=10, color='color', legend='label', source=s)
        show(p)


if __name__ == '__main__':
    files = ["gen_activ.p", "real_activ.p"]
    dirs = ["./generated", "./real"]
    labels = ["Generated", "Real"]
    tsb = tsneBokeh(dirs, 1000, files, labels)
    tsb.createPlot()
