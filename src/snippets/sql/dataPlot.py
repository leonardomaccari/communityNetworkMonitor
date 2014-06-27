
import matplotlib.pyplot as plt
import sys

#FIXME improve the comments in this file

class dataPlot:
    """ Simple class to serialize the generation of graphs. C
    is a configuration class I use in get_stats_dump, it is not
    needed """

    def __init__(self, C=None, extension=".png"):
        """ outfile is where the plot will be saved, with the specified
        image extension"""
        self.x = []
        self.y = []
        self.title = ""
        self.xAxisLabel = ""
        self.yAxisLabel = ""
        self.outFile = ""
        self.key = []
        self.legendPosition = "center right"
        if C == None:
            self.fileType = extension
        else:
            self.fileType = "."+C.imageExtension

    def plotData(self, style = "-", bw="COLOR"):
        """ plot the data and save the result in a graphic file """

        if self.outFile == "":
            return
        dataDimension = 0
        ax = plt.subplot(111)
        for y in self.y:
            l = self.y[dataDimension][1]
            v = self.y[dataDimension][0]
            if l != "": 
                ax.plot(self.x[dataDimension],
                    v, style, label=l)
            else :
                ax.plot(self.x[dataDimension], v, style)
            dataDimension += 1
        if bw == "BW":
            self.setAxLinesBW(ax, bw=True)
        elif bw == "PRINTABLE":
            self.setAxLinesBW(ax, bw=False)

        plt.title(self.title)
        plt.xlabel(self.xAxisLabel)
        plt.ylabel(self.yAxisLabel)
        if self.legendPosition == "aside":
            box = ax.get_position()
            ax.set_position([box.x0, box.y0,
                                 box.width * 0.8, box.height])
            ax.legend(loc="center left", fancybox=True, 
                bbox_to_anchor=(1, 0.5), shadow=True, 
                prop={'size':15}, numpoints=1)
        else: 
            plt.legend(loc=self.legendPosition, fancybox=True, 
                shadow=True, numpoints=1)
        plt.savefig(self.outFile+self.fileType)
        plt.clf()


    def saveData(self):
        """ save a text file with the data """
        if self.outFile == "":
            return
        dataDimension = max([len(v) for v in self.x])
        maxDimension = sorted(self.x,  key=lambda x: len(x))[-1]
        outFileName = self.outFile+".txt"
        try:
            outFile = open(outFileName, "w")
        except IOError:
            print >> sys.stderr, "ERROR: could not open the txt output file",\
                outFileName

        print >> outFile, self.xAxisLabel.ljust(10),

        for l in self.y:
            print  >> outFile, l[1].ljust(10),
        print >> outFile 
        counter = 0
        for counter in range(dataDimension):
            print  >> outFile, str(maxDimension[counter])+",".ljust(10), 
            for l in self.y:
                if len(l[0]) > counter:
                    print  >> outFile, str(l[0][counter])+",".ljust(10),
                else:
                    print  >> outFile, ",".ljust(10),

            print >> outFile
        outFile.close()


    
    ### following code comes from http://stackoverflow.com/questions/7358118/matplotlib-black-white-colormap-with-dashes-dots-etc
    def setAxLinesBW(self, ax, bw=True):
        """
        Take each Line2D in the axes, ax, and convert the line style to be 
        suitable for black and white viewing. bw=False still uses colors
        """
        print ax, type(ax)
        MARKERSIZE = 3
    
        COLORMAP = {
            'b': {'marker': None, 'dash': (None,None)},
            'g': {'marker': None, 'dash': [10,10]},
            'r': {'marker': None, 'dash': [10,6,2,6]},
            'c': {'marker': None, 'dash': [2,6]},
            'm': {'marker': None, 'dash': [10,4,10,4,10,20]},
            'y': {'marker': None, 'dash': [10,6,2,4,2,20]},
            'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
            }
    
        for line in ax.get_lines():# + ax.get_legend().get_lines():
            origColor = line.get_color()
            if bw:
                line.set_color('black')
            line.set_dashes(COLORMAP[origColor]['dash'])
            line.set_marker(COLORMAP[origColor]['marker'])
            line.set_markersize(MARKERSIZE)
    
    def setFigLinesBW(self, fig):
        """
        Take each axes in the figure, and for each line in the axes, make the
        line viewable in black and white.
        """
        for ax in fig.get_axes():
            self.setAxLinesBW(ax)
