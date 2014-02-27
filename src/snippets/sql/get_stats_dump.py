#!/usr/bin/python

import sys
import os
from sqlalchemy import create_engine
from collections import defaultdict
from sqlalchemy.orm import sessionmaker, scoped_session
import code
import numpy as np
import networkx as nx
import matplotlib
# use the Agg backend, not X, for ssh shells
matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = 12, 12 
matplotlib.rcParams['figure.dpi'] = 1200
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import optimize
import cPickle as pk
import getopt
from datetime import datetime
import shutil
from multiprocessing import Process, Queue
from scipy import stats
from copy import copy

# these module-level functions are needed or else I can 
# not pickle the data structure
def dd1():
    return defaultdict(list)
def dd2():
    return defaultdict(dd1)
def dd3():
    return defaultdict(dd2)
def dd4():
    return defaultdict(dd3)
def dd5():
    return defaultdict(dd4)


class dataObject:
    def __init__(self):
        self.scanTree = dd2()
        self.rawData = dd2()
        self.dataSummary = dd3()
        self.dumpFile = ""
        self.routeData = dd5()
        self.etxThreshold = -1

    def initialize(self, fileName):
        self.dumpFile = fileName 
        try:
             f = open(self.dumpFile, "r")
        except IOError:
             print "could not load",self.dumpFile
             raise 
        d = pk.load(f)
        self.scanTree = d.scanTree
        self.rawData = d.rawData
        self.dataSummary = d.dataSummary
        self.routeData = d.routeData
        f.close()
    def save(self, fileName):
        f = open(fileName, "w")
        pk.dump(self,f)
        f.close()
    def printSummary(self):
        logString = ""
        for net in self.scanTree:
            logString += "===== "+net+" =====\n"
            logString += "scans: " + str(len(self.dataSummary[net])) 
            logString += "\n"
            logString += "\n"
            # print the header
            for sid in data.dataSummary[net]:
                logString += str("ID").ljust(4)
                for label in data.dataSummary[net][sid]:
                    logString += label[0].ljust(label[1]) + " "
                logString += "\n"
                break
            # print the data
            for sid in sorted(data.dataSummary[net]):
                logString += str(sid).ljust(4)
                for key, value in data.dataSummary[net][sid].items():
                    logString += str(value).ljust(key[1]) + " "
                logString += "\n"
        logString += "\n\nETX threshold:" + str(self.etxThreshold)

        return logString

class dataParser():
    net = "" 
    q = None


    def __init__(self, netName, queue):
        self.net = netName
        self.q = queue
        self.weightStats = dd1()

    def run(self, rData, routeData, sData):
	""" I need to pass the data here and not in the constructor
	    or else I duplicate the whole memory of the father"""
	self.rawData = rData
	self.routeData = routeData
	self.dataSummary = sData
	print "data loaded", self.net
        try:
            os.makedirs(C.resultDir+self.net)
        except OSError:
            try:
                os.mkdir(C.resultDir+"/backup/")
            except:
                pass
            now =  datetime.now()
            newFolder = C.resultDir+"backup/"+self.net+str(now.month)+\
                    str(now.day)+str(now.hour)+\
                    str(now.minute)+str(now.second)
            shutil.move(C.resultDir+self.net, newFolder)
            os.mkdir(C.resultDir+self.net)
        netEtx = []
        retValue = {}
        for scanId in self.rawData:
            netEtx += self.rawData[scanId]
        self.getRouteDistributions(self.net)
        self.getDegreeDistribution(self.net)
        bins = np.array(range(1,1000))/100.0
        retValue["etx"] = self.getETXDistribution(netEtx, bins)
        
        retValue["link"] = self.getLinkDistributions(self.net)
        retValue["ROBUSTNESS"] = self.getRobustness()
        q.put(retValue)

    def getRobustness(self):
        coreRobustness = []
        averageRobustness = defaultdict(list) 
        averageCoreRobustness = defaultdict(list) 
        for scanId in self.routeData:
            G = self.routeData[scanId]["Graph"]
            r = self.computeRobustness(G, tests=30)[0]
            for k,v in r.items():
                percent = int(100*float(k)/len(G.edges()))
                averageRobustness[percent].append(v)
                #x.append(float(k)/len(G.edges()))
            r = self.computeRobustness(G, tests=30, mode="core")[0]
            for k,v in r.items():
                percent = int(100*float(k)/len(G.edges()))
                averageCoreRobustness[percent].append(v)
                coreRobustness.append(v)
        retValue = defaultdict(dict)
        retValue["RB"]["x"] = sorted(averageRobustness.keys())
        retValue["RB"]["y"] = [np.average(averageRobustness[k]) \
                for k in sorted(averageRobustness.keys())]
        retValue["CRB"]["x"] = sorted(averageRobustness.keys())
        retValue["CRB"]["y"] = [np.average(averageCoreRobustness[k]) \
                for k in sorted(averageCoreRobustness.keys())]
        return retValue



    def getETXDistribution(self, etxList, b):
        cleanList = [e for e in etxList if e < 10]
        h,b = np.histogram(cleanList, bins=b) 
        totalSamples = sum(h)*1.0
        etx = plt.subplot(1,1,1)
        etx.yaxis.tick_left()
        etx.plot(b[:-1], h, color="blue")
        plt.ylabel("samples")
    
        sumSamples = []
        partialSum = 0
        for v in h:
            partialSum += v
            sumSamples.append(partialSum)
        etxSum=etx.twinx()
        etxSum.yaxis.tick_right()
        etxSum.yaxis.set_label_position("right")
        etxSum.set_ylim([0,1])
        etxSum.plot(b[:-1], 
                (np.array(sumSamples)/totalSamples), 
                color="red")
        plt.title("ETX value frequency/cumulative, "+self.net)
        
        ETXDistribution = {}
        ETXDistribution["x"] = b[:-1]
        ETXDistribution["y"] = (np.array(sumSamples)/totalSamples)
            
        etxFolder = C.resultDir+self.net+"/ETX/"
    
        try:
            os.mkdir(etxFolder)
        except:
            pass
        plt.savefig(etxFolder+"ETX-distribution."+C.imageExtension)
        plt.clf()
        return ETXDistribution
    
    
    def getLinkDistributions(self, net):
        linkFolder = C.resultDir+net+"/LINKS/"
        try:
            os.mkdir(linkFolder)
        except:
            pass
        etxRanking = []
        linkDict = defaultdict(list)
        for scanId in self.routeData:
            g = self.routeData[scanId]["Graph"]
            for s,d,etxDict in g.edges(data=True):
                # cleaning meanless links
                if etxDict["weight"] < 20:
                    linkDict[(s,d)].append(etxDict["weight"])

        for link, etxArray in linkDict.items():
            # considering only the links that appear in 
            # at least 30% of the samples
            if len(etxArray) < 0.3*len(self.dataSummary):
                continue
            avgEtx = np.average(etxArray)
            avgStd = np.std(etxArray)
            etxRanking.append([avgEtx, avgEtx+avgStd/2,\
                avgEtx-avgStd/2])
         
        x = range(len(etxRanking))
        avg = [e[0] for e in sorted(etxRanking, key = lambda x: x[0])]
        sup = [e[1] for e in sorted(etxRanking, key = lambda x: x[0])]
        inf = [e[2] for e in sorted(etxRanking, key = lambda x: x[0])]

        plt.plot(x, avg, x, sup, x,  inf)
    
        plt.title("Average and standard deviation for ETX per link")
        plt.xlabel("link Id")
        plt.ylabel("Average ETX +/- 0.5*std")
        plt.savefig(linkFolder+"/link-ranking."+C.imageExtension)
        plt.clf()
        linkRanking = {}
        linkRanking["x"] = x
        linkRanking["avg"] = avg
        linkRanking["sup"] = sup
        linkRanking["inf"] = inf
        return linkRanking
    
    def getRouteDistributions(self, net):
        routeFolder = C.resultDir+net+"/ROUTES"
        try:
            os.mkdir(routeFolder)
        except:
            pass
        b = np.array(range(1,201))/10.0
        numHops = []
        etxList = []
        etxTime = dd2()
        numHopMap = defaultdict(list)
        for scan in self.routeData:
            D = self.routeData[scan]['data']
            for s in D:
                for d in D[s]:
                    if D[s][d][1] < 60:
                        numHops.append(D[s][d][0])
                        etxList.append(D[s][d][1])
                        numHopMap[D[s][d][0]].append(D[s][d][1])
                        etxTime[scan][D[s][d][0]].append(D[s][d][1])
        etxAvg = dd1()
        # scan Ids are incremental with time
        for scan in sorted(etxTime):
            for length in etxTime[scan].keys():
                avgArray = etxTime[scan][length]
                etxAvg[length].append(np.average(avgArray))


        for length in  sorted(etxAvg):
            plt.plot(etxAvg[length])
        plt.title("Route Variation, "+net)
        plt.xlabel("snapshot id")
        plt.ylabel("ETX weight")
        plt.savefig("/tmp/"+net+"routeTimeVariation."+C.imageExtension)
        plt.clf()

        maxHops = int(max(numHops))
        maxETX = int(max(etxList)+1) # the upper integer
        hh,bh = np.histogram(numHops, bins=range(maxHops))
        he,be = np.histogram(etxList, bins=range(maxETX))
        plt.xlabel("ETX weight/hop count")
        freq = plt.subplot(1,1,1)
        cum = freq.twinx()
        cum.yaxis.tick_right()
        cum.yaxis.set_label_position("right")
        cum.set_ylim([0,1])
        #plt.legend([ep,hp], ["etx", "hopCount"])
        freq.legend(loc="center right")
        cumulativeH = []
        cumulativeE = []
        partialsum = 0.0
        plt.title("Frequency of route length and weight, "+net)
        for i in he:
            partialsum += i
            cumulativeE.append(partialsum)
    
        cum.plot(be[:len(cumulativeE)], 
                np.array(cumulativeE)/partialsum, ls="steps--")
        freq.plot(be[:len(he)], np.array(he)/partialsum, label="weight", drawstyle="steps")
        freq.set_xticks(np.array(be[1:])-0.5)
        freq.set_xticklabels(be[1:], fontsize=16 )
        partialsum = 0.0
        totRoutes = 0
        for i in hh:
            partialsum += i
            cumulativeH.append(partialsum)
            totRoutes = partialsum
        cum.set_xlim([0,25])
        cum.plot(bh[:len(cumulativeH)], 
                np.array(cumulativeH)/partialsum, ls="steps--")
        freq.plot(bh[:len(hh)], np.array(hh)/totRoutes, label="length", drawstyle="steps")
        freq.xaxis.grid(linestyle='--', linewidth=1, alpha=0.1)
        plt.savefig(routeFolder+"/routes."+C.imageExtension)
    

        plt.clf()
        sortedEtx = []
        for l in sorted(numHopMap):
            sortedEtx.append(numHopMap[l])
        labels = []
        freqLabels = []
    
        weightDistributions = []
    
        bins = np.array(range(0,250,1))/float(10)
        for l in sorted(numHopMap):
            hw, bw = np.histogram(numHopMap[l], bins=bins)
            weightDistributions.append(hw)
            frac = 100*len(numHopMap[l])/float(totRoutes)
            if int(frac) > 0 :
                rFrac = str(int(100*len(numHopMap[l])/float(totRoutes)))
            else:
                rFrac = "."+str(int(frac*10))

            labels.append(str(l) + "\n" + str(rFrac) + "%")
            freqLabels.append(str(rFrac) + "%")
    
        ax1 = plt.subplot(111)
        ax1.set_xticklabels(labels, fontsize = 14)
        ax1.boxplot(sortedEtx)
        ax1.set_ylim([0,40])
        plt.title("Route weight Vs route length")
        plt.savefig(routeFolder+"/boxplot."+C.imageExtension)
        plt.clf()
        
        i = 0
        for h in weightDistributions:
            i += 1
            plt.plot(bins[:-1], h, label="length "+str(i))
        plt.title("Distribution of the route weights")
        plt.xlabel("Route weight (ETX)")
        plt.ylabel("samples")
        plt.legend(loc="upper right", ncol=2)
        plt.savefig(routeFolder+"/etxWeights."+C.imageExtension)
        plt.clf()
    
    def getDegreeDistribution(self, net):
        routeFolder = C.resultDir+net+"/ROUTES"
        degreeDistribution = defaultdict(list)
        degreeDistributionNL = defaultdict(list)
        for scanId in self.routeData:
            graph = self.routeData[scanId]["Graph"]
            for node in graph:
                if len(graph[node]) > 1:
                    degreeDistributionNL[node].append(len(graph[node]))
                    degreeDistribution[node].append(len(graph[node]))
                else:
                    degreeDistribution[node].append(len(graph[node]))

        degFloat = [int(round(np.average(degreeDistribution[v]),0)) for \
                v in degreeDistribution]
        bins = range(1,int(max(degFloat))+1)
        h,b = np.histogram(degFloat, bins = bins)

        degNLFloat= [np.average(degreeDistributionNL[v]) for \
                v in degreeDistributionNL]
        bins = range(1,int(max(degNLFloat))+1)
        hNL,bNL = np.histogram(degNLFloat, bins = bins)


        x = []
        y = []
        for i in range(len(h)):
            if h[i] > 0:
                y.append(h[i])
                x.append(b[i])
        xNL = []
        yNL = []
        for i in range(len(hNL)):
            if hNL[i] > 0:
                yNL.append(hNL[i])
                xNL.append(bNL[i])
        fitfunc = lambda p, x: p[0] * x ** (p[1])
        errfunc = lambda p, x, y: (y - fitfunc(p, x))
        rcParams.update({'font.size': 35})
        out,success = optimize.leastsq(errfunc, 
                [1,-1],args=(x,y))
        fittedValue = []
        for v in x:
            fittedValue.append(out[0]*(v**out[1]))
        p = plt.subplot(1,1,1)
        p.plot(x, y, "bo", x, fittedValue, markersize=15)
        plt.title("Degree relative frequency (%)")
        plt.savefig(routeFolder+"/degree."+C.imageExtension)
        plt.clf()

        p = plt.subplot(1,1,1)
        p.plot(xNL, yNL, "bo", markersize=15)
        p.set_xscale("log")
        p.set_yscale("log")
        plt.title("Degree relative frequency, non-leaf subgraph")
        plt.savefig(routeFolder+"/NLdegree."+C.imageExtension)
        plt.clf()
        rcParams.update({'font.size': 20})
    

    def diffVectors(self, v1,v2):
        if len(v1) != len(v2):
            print >> sys.stderr, "Error, comparing two different arrays", v1, v2
            #sys.exit(1)
            #FIXME this happens sometimes
        diff = 0
        for item in v1:
            if item not in v2:
                diff += 1
        return diff
    
    
    def computeRobustness(self, graph, tests=100, mode="simple"):

        links = []
        weights = []
        for l in graph.edges(data=True):
            if mode == "core":
                if len(graph[l[0]]) != 0 and \
                        len(graph[l[1]]) != 0:
                    links.append((l[0], l[1])) 
            else:
                links.append((l[0], l[1])) 
            weights.append(float(l[2]["weight"]))

        totWeight = sum(weights)
        normalizedWeight = [l/totWeight for l in weights]
        custDist = stats.rv_discrete(values=(range(len(links)),
            normalizedWeight), name="custDist")

        mainCSize = defaultdict(list)
        mainNonCSize = defaultdict(list)
        nlen = float(len(graph.nodes()))
        elen = float(len(graph.edges()))
        for i in range(tests):
            purgedGraph = graph.copy()
            purgedLinks = []
            for k in range(1,int(elen/2)):
                r = custDist.rvs()
                if len(purgedLinks) >= elen:
                    print >> sys.stderr, "Trying to purge",k,"links",\
                        "from a graph with",elen," total links" 
                    break
                while (r in purgedLinks):
                    r = custDist.rvs()
                purgedLinks.append(r) 
                l = links[r]
                purgedGraph.remove_edge(l[0],l[1])
                compList =  nx.connected_components(purgedGraph)
                mainCSize[k].append(len(compList[0])/nlen)
		
                compSizes = [len(r) for r in compList[1:]]
                if len(compSizes) == 0:
                    mainNonCSize[k].append(0)
                else:
                    mainNonCSize[k].append(
                        np.average([len(r) for r in compList[1:]])/nlen)
        mainCSizeAvg = {}
        for k, tests in mainCSize.items():
            mainCSizeAvg[k] = np.average(tests)
        return mainCSizeAvg, mainNonCSize
        

class dataPlot:

    def __init__(self, C=None):
        self.x = []
        self.y = []
        self.title = ""
        self.xAxisLabel = ""
        self.yAxisLabel = ""
        self.outFile = ""
        self.key = []
        self.legendPosition = "center right"
        if C == None:
            self.fileType = ".png"
        else:
            self.fileType = "."+C.imageExtension

    def plotData(self, style = "-"):
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


def extractDataSeries(retValues):

    etx = dataPlot(C)
    link = dataPlot(C)
    mprLq = dataPlot(C)
    mprRFC = dataPlot(C)
    betweenness = dataPlot(C)
    betweennessD = dataPlot(C)
    closeness = dataPlot(C)
    mprRobustness = dataPlot(C)
    closenessV = dataPlot(C)
    robustness = dataPlot(C)
    comparisonFolder = C.resultDir+"/COMPARISON/"
    singleNodeCloseness = dataPlot(C)
    singleNodeBetweenness = dataPlot(C)
    MPRSigHistogram = defaultdict(list)

    for n,v in retValues.items():
        if "etx" in v:	
            etx.y.append((v["etx"]["y"], n))
            etx.x.append(v["etx"]["x"])
            etx.title = "ETX ECDF"
            etx.outFile = comparisonFolder+"etx"
            etx.xAxisLabel = "ETX"
            etx.yAxisLabel = ""
            etx.legendPosition = "center right"

        if "link" in v:	
            link.x.append(v["link"]["x"])
            link.y.append((v["link"]["avg"], n))
            relStddList = []
            for i in range(len(v["link"]["avg"])):
                relStddList.append((v["link"]["sup"][i] - 
                        v["link"]["inf"][i])/
                    v["link"]["avg"][i])
            link.outFile = comparisonFolder+"link"
            link.title  = "Average ETX per link with stddev"
            link.xAxisLabel = "link"
            link.yAxisLabel = "ETX"
            link.legendPosition = "center left"

        if "ROBUSTNESS" in v:
            robustness.x.append(v["ROBUSTNESS"]["RB"]["x"])
            robustness.y.append((v["ROBUSTNESS"]["RB"]["y"],n))
            robustness.x.append(v["ROBUSTNESS"]["CRB"]["x"])
            robustness.y.append((v["ROBUSTNESS"]["CRB"]["y"],n+" core"))
            robustness.title = "Robustness metrics"
            robustness.xAxisLabel = "Failed links (%)"
            robustness.legendPosition = "lower left"
            robustness.outFile = comparisonFolder+"graphrobustness"

    

    etx.plotData()
    link.plotData()
    mprLq.plotData()
    mprRFC.plotData()
    mprRobustness.plotData()
    closeness.plotData()
    betweenness.plotData()
    betweennessD.plotData()
    closenessV.plotData()
    robustness.plotData()
    singleNodeCloseness.plotData()
    singleNodeBetweenness.plotData()
 
    
    if len(MPRSigHistogram) != 0:
        ax = plt.subplot(111)
        shift = 0.3
        width = 0.3
        x = np.array(range(1,len(MPRSigHistogram["RFC"])+1))
        ax.bar(x, MPRSigHistogram["RFC"], width, color = "b", label="RFC")
        ax.bar(shift + x, MPRSigHistogram["LQ"], width, color = "r", label="lq")
        ax.bar(2*shift + x, MPRSigHistogram["WC"], width, color = "g", 
                label="LSR")
        ax.set_xticks(x+shift)
        lb = ax.set_xticklabels(MPRSigHistogram["label"])
        plt.setp(lb,rotation=45)
        plt.legend(loc="upper left")
        plt.title("TC messages per link per TC emission interval")
        plt.savefig(comparisonFolder+"signallingHist."+C.imageExtension)
        plt.clf()




class configuration:
    def __init__(self):
        self.loadFile = ""
        self.loadDb = ""
        self.numRuns = None
        self.saveDump = ""
        self.resultDir = "/tmp/CN/"
        self.etxThreshold = 10 # filter out links with etx larger than this
        self.printInfo = False
        self.maxGroupSize = 5
        self.createTestRun = False
        self.imageExtension = "png"
    def checkCorrectness(self):
        if self.loadFile != "" and self.loadDb != "":
            print "Error: You can not specify both file and db to load" 
            self.usage()
            return False
        if self.printInfo and self.loadFile == "":
            print "Error: Please sepcify a pickle file to dump info"
            self.usage()
            return False
        if self.createTestRun == True and self.loadDb == "":
            print "Error: please specify a db to load for a shrinked testfile"
            self.usage()
            return False
        if self.numRuns == None:
            self.numRuns = -1
        return True
    def usage(self):
        print "usage:"
        print "-d database" 
        print "-f [file] pickled dump file from previous run" 
        print "-s [file] save new database pickle file"
        print "-S [file] save new database pickle file (reducing graphs ",\
                "to less than 40 nodes, testing only)"
        print "-p print pickle file summary"
        print "-r number of runs to consider when saving pickle file"
        print "-v use vector graphics for output (eps)"

def createFolder(folder):
    f = C.resultDir+"/"+folder
    try:
        os.makedirs(f)
    except OSError:
        try:
            os.mkdir(C.resultDir+"/backup/")
        except:
            pass
        now =  datetime.now()
        newFolder = C.resultDir+"backup/"+folder+str(now.month)+\
                str(now.day)+str(now.hour)+\
                str(now.minute)+str(now.second)
        shutil.move(C.resultDir+"/"+folder, newFolder)
        os.mkdir(C.resultDir+"/"+folder)



## global configuration class and data
C = configuration()
data = dataObject()

if  __name__ =='__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:f:r:s:pS:v")
    except getopt.GetoptError, err:
        # print help information and exit:
        print >> sys.stderr,  str(err)
        C.usage()
        sys.exit(2)
    for option,v in opts:
        if option == "-f":
            C.loadFile = v
            continue
        if option == "-d":
            C.loadDb = "sqlite:///" + v
            continue
        if option == "-r":
            C.numRuns = int(v)
            continue
        if option == "-s":
            C.saveDump = v
            continue
        if option == "-p":
            C.printInfo = True
            continue
        if option == "-S":
            C.createTestRun = True
            C.saveDump = v
            continue
        if option == "-v":
            C.imageExtension = "eps"
            continue
    if not C.checkCorrectness():
        sys.exit(1)


    rcParams.update({'font.size': 20})
    startTime =  datetime.now()
    print "loading", datetime.now()
    print C.loadDb
    if C.loadFile != "":
        try:
            data.initialize(C.loadFile)
        except IOError:
            print "could not read data file"
            sys.exit(1)
        if C.printInfo:
            print data.printSummary()
            sys.exit()


    loadTime =  datetime.now() - startTime
    if C.saveDump != "":
        data.save(C.saveDump)
        sys.exit()

    logString = ""
    print "loaded", datetime.now()

    createFolder("CENTRALITY")
    createFolder("COMPARISON")
    parsers = []
    for net in data.rawData:
        q = Queue()
        parser = dataParser(net, q)
        p = Process(target=parser.run, args = (data.rawData[net], 
    		data.routeData[net], 
	    	data.dataSummary[net]))
        data.rawData[net] = {}
        data.routeData[net] = {}
        data.dataSummary[net] = {}
        parsers.append((net, p, q))
        p.start()

    retValues = defaultdict(dict)
    while True:
        alive = len(parsers)
        for (n,p,q) in parsers:
        	# a process doesn't die if its queue is
        	# not emptied
                if not q.empty():
        	    retValues[n] = q.get()
        	    print "Subprocess", n, "exited"
        	if not p.is_alive():
        	    alive -= 1
        	
        if alive == 0:
            break
        import time
        time.sleep(1)
    extractDataSeries(retValues)

    f = open(C.resultDir+"/logfile.txt", "w")
    runTime = datetime.now() - startTime - loadTime
    print >> f,  data.printSummary()
    print >> f, logString
    print >> f, "Whiskers = q3+/-1.5*IQR"
    print >> f, "loadTime =", loadTime, "runTime =", runTime
    f.close()


