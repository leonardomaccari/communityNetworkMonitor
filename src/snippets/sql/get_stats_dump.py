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
matplotlib.rcParams['figure.figsize'] = 20, 20 
matplotlib.rcParams['figure.dpi'] = 300
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
import simplejson

sys.path.append("../../")
sys.path.append("dataPlot/")
sys.path.append("community_networks_analysis/")

from myCrypto import myCrypto

try:
    from dataplot import dataPlot
    from groupmetrics import computeGroupMetrics, groupMetricForOneGroup
    from mpr import solveMPRProblem, purgeNonMPRLinks
    from robustness import computeRobustness
    from routecomparison import * 
except ImportError, m:
    print >> sys.stderr, """ERROR: You are missing needed submodules:
    """, m, """
    please run :
        git submdule init
        git submodule update
        """
    sys.exit()

# WARNING: this is a collection of functions i used to 
# produce the results for the Elsevier Ad-Hoc networks journal
# it is not well documented, not particularly easy to understand, 
# but it can give inputs to extract data from the community network
# database. Take a look at the dataParser() function, that is the one that calls
# the functions that do the real data analysis.


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
    """ This class is used to store and load pickle files """
    def __init__(self):
        self.scanTree = dd2()
        self.rawData = dd2()
        self.dataSummary = dd3()
        self.dumpFile = ""
        self.routeData = dd5()
        self.etxThreshold = -1
        self.namesDictionary = {}

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

        if C.namesDictionaryFileName != "":
            try:
                namesFile = open(C.namesDictionaryFileName, "r")
            except IOError:
                print "ERROR: Could not load the name dictionary file",\
                    C.namesDictionaryFileName
                return
            try:
                self.namesDictionary =  simplejson.load(namesFile)
            except simplejson.JSONDecodeError:
                print "ERROR: could not parse the JSON names file"
            if not namesFile.closed:
                namesFile.close()
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

def dumpGraphs(data, graphNumber):
    """ this function dumps in the /tmp/ folder a number of graphs in matrix 
    format """

    for net in data.routeData:
        scanIdList = data.routeData[net].keys()
        for i in range(graphNumber):
                if i < len(scanIdList):
                    g = data.routeData[net][scanIdList[i]]["Graph"]
                    relabels = {}
                    for node in g.nodes():
                        relabels[node] = g.nodes().index(node)
                    nx.relabel_nodes(g, relabels, copy=False)
                    nx.write_edgelist(g,"/tmp/"+net+str(i)+".edges", 
                            data=["weight"])


def getDataSummary(ls, data):
    """ this is the function i use to extract the data from the database and 
    populate the pickle files """

    scanQuery = "SELECT * from scan"
    QUERY="""select snode.Id AS sid, snode.owner_Id AS soid, sperson.username AS
           sname, sperson.email AS semail, dnode.Id AS did, dnode.owner_Id as doid,
           dperson.username AS dname, dperson.email AS demail, etx.etx_value AS etxv 
           from link, scan, node as snode, node as dnode, etx, person AS sperson,
           person as dperson \
           WHERE link.scan_Id = scan.Id AND snode.Id = link.from_node_Id \
           AND dnode.Id = link.to_node_Id AND etx.link_Id = link.Id \
           AND dnode.scan_Id = scan.Id AND snode.scan_Id = scan.Id 
           AND scan.Id= %d AND soid = sperson.Id AND doid = dperson.Id"""

    try:
        q = ls.query("Id", "time", "scan_type", "network", "encrypted")\
                .from_statement(scanQuery)
        if len(q.all()) == 0:
            raise
    except:
        print "something went wrong with opening the db"
        sys.exit(1)

    numScan = len(q.all())
    scanCounter = 0
    data.etxThreshold = C.etxThreshold
    for [scanId, scanTime, scanType, scanNetwork, encrypted] in q:
        data.scanTree[scanNetwork][scanType].append([scanId, scanTime, encrypted])
        print encrypted, scanId

    
    for net in data.scanTree:
        counter = 0
	    # for graz I have one sample every 10 minutes,
        # for ninux/Wien I have one sample every 5 minutes
        if net == "FFGraz":
            networkPenalty = 2
        else:
            networkPenalty = 1
        for scanId in data.scanTree[net]['ETX']:
            if C.numRuns > 0 and counter >= C.numRuns/networkPenalty:
                break
            queryString = QUERY % scanId[0]
            q = ls.query("sid", "soid", "sname", "semail", "did",
                    "doid", "dname", "demail", "etxv").\
                    from_statement(queryString)
            dirtyG = nx.Graph()
            noValue = "UNKNOWN"
            for values in q:
                sid = C.myCrypto.decrypt(values[0])
                did = C.myCrypto.decrypt(values[4])
                if values[2] != noValue:
                    sname = C.myCrypto.decrypt(values[2])
                else:
                    sname = noValue
                if values[3] != noValue:
                    semail = C.myCrypto.decrypt(values[3])
                else:
                    semail = noValue

                if values[6] != noValue:
                    dname = C.myCrypto.decrypt(values[6])
                else:
                    dname = noValue

                if values[7] != noValue:
                    demail = C.myCrypto.decrypt(values[7])
                else:
                    demail = noValue

                dirtyG.add_node(sid, owner_Id = values[1], username=sname,
                        email=semail)
                dirtyG.add_node(did, owner_Id = values[5], username=dname,
                        email=demail)
                if values[8] < C.etxThreshold:
                    dirtyG.add_edge(sid, did, weight=float(values[8]))

            if C.createTestRun == True:
                nd = dirtyG.degree()
                for n in sorted(nd.items(), key = lambda x: x[1]):
                    dirtyG.remove_node(n[0])
                    if len(dirtyG) < 40:
                        break

            if len(dirtyG) != 0:
                G = nx.connected_component_subgraphs(dirtyG)[0]
                componentSize = len(G)
            else:
                G = nx.Graph()
                componentSize = 0
            if componentSize < 10:
                continue
            counter += 1

            etxV = [e[2]['weight'] for e in G.edges(data=True)]
            data.rawData[net][scanId[0]] = etxV
            data.routeData[net][scanId[0]]["Graph"] = G
            weightedPaths = nx.shortest_path(G, weight="weight")
            for s in G.nodes():
                for d in G.nodes():
                    if s == d:
                        continue
                    if d in data.routeData[net][scanId[0]]["data"] and \
                            s in data.routeData[net][scanId[0]]["data"][d]:
                        continue
                    currPath = weightedPaths[s][d]
                    pathWeight = 0
                    for i in range(len(currPath) - 1):
                        pathWeight += G[currPath[i]][currPath[i+1]]["weight"]
                    data.routeData[net][scanId[0]]["data"][s][d] = [len(weightedPaths[s][d])-1, 
                            pathWeight]
            data.routeData[net][scanId[0]]["Graph"] = G


            nd = filter(lambda x: x==1, dirtyG.degree().values())
            nl = len(nd)
            nn = len(dirtyG)
            le = len(etxV)
            avgEtx = np.average(etxV)
            data.dataSummary[net][scanId[0]][("numLeaves", 9)] = nl
            data.dataSummary[net][scanId[0]][("time", 30)] = scanId[1]
            data.dataSummary[net][scanId[0]][("numNodes",9)] = nn
            data.dataSummary[net][scanId[0]][("numEdges",9)] = le
            data.dataSummary[net][scanId[0]][("largestComponent",16)] = \
                    componentSize
            data.dataSummary[net][scanId[0]][("etxAvg",8)] = str(avgEtx)[0:5]

            scanCounter += 1
            if int((100000 * 1.0*scanCounter / numScan)) % 10000 == 0:
                print int((100 * 1.0*scanCounter / numScan)),"% complete"

class dataParser():
    """ each network has its own dataparser, each dataparser instance runs in a
    separate process to speed up computation """

    # FIXME using processes requires a lot of memory if you want to run them in
    # parallel

    net = "" 
    q = None

    def __init__(self, netName, queue):
        self.net = netName
        self.q = queue
        self.weightStats = dd1()

    def run(self, rData, routeData, sData, namesDictionary):
	""" I need to pass the data here and not in the constructor
	    or else I duplicate the whole memory of the father"""

        self.rawData = rData
        self.routeData = routeData
        self.dataSummary = sData
        self.namesDictionary = namesDictionary
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

        #### here you can enable the single functions you want to run

        #retValue["degree"] = self.getDegreeDistribution(self.net)
        #self.routeComparison()
        #bins = np.array(range(1,1000))/100.0
        #retValue["etx"] = self.getETXDistribution(netEtx, bins)
        #retValue["link"] = self.getLinkDistributions(self.net)
        #retValue["CENTRALITY"] = self.getCentralityMetrics(self.net)
        retValue["MPRRFC"] = self.getMPRSets(self.net, "RFC")
        retValue["MPRLQ"] = self.getMPRSets(self.net, "lq")
        retValue["ROBUSTNESS"] = self.getRobustness()
        q.put(retValue)
   
    def getRobustness(self):
        """ compute network robustness """
        coreRobustness = []
        averageRobustness = defaultdict(list) 
        averageCoreRobustness = defaultdict(list) 
        for scanId in self.routeData:
            G = self.routeData[scanId]["Graph"]
            r = computeRobustness(G, tests=30)[0]
            for k,v in r.items():
                percent = int(100*float(k)/len(G.edges()))
                averageRobustness[percent].append(v)
                #x.append(float(k)/len(G.edges()))
            r = computeRobustness(G, tests=30, mode="core")[0]
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
        """ compute the distribution of the ETX (link weights) """
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
            
        #etxSum.set_yscale("log")
        #etxSum.set_xscale("log")
        #etx.set_xscale("log")
        #etx.set_yscale("log")
        etxFolder = C.resultDir+self.net+"/ETX/"
    
        try:
            os.mkdir(etxFolder)
        except:
            pass
        plt.savefig(etxFolder+"ETX-distribution."+C.imageExtension)
        plt.clf()
        return ETXDistribution
    
    
    def getLinkDistributions(self, net):
        """ get temporal average of the link weights """
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
        """ compute the average lenght and weight for the shortest routes """
        routeFolder = C.resultDir+net+"/ROUTES"
        try:
            os.mkdir(routeFolder)
        except:
            pass
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
        #etxStd = []
        #etxMap = defaultdict(list)
        #printDists = -1
        #for s in etxTime:
        #    for d in etxTime[s]:
        #        e = zip(*etxTime[s][d])[0]
        #        c = zip(*etxTime[s][d])[1]
        #        interval = max(e)-min(e)
        #        etxStd.append(interval)
        #        etxMap[round(np.average(c))].append(interval)
        #        if printDists > 0:
        #            plt.plot(e)
        #            plt.savefig(routeFolder+"/ETXd-"+s+"-"+d+".eps")
        #            plt.clf()
        #            printDists -= 1
    
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
                np.array(cumulativeE)/partialsum, 
                color="blue", ls="steps--", linewidth=6)
                #label = "Cumulative ETX")
        freq.plot(be[:len(he)], np.array(he)/partialsum, 
                label="weight", drawstyle="steps", color="blue",
                linewidth=6)
        freq.set_xticks(np.array(be[1:])-0.5)
        labels = []
        for l in be[1::2]:
            labels += [l]
            labels += [""]
        freq.set_xticklabels(labels, fontsize=40 )
        partialsum = 0.0
        totRoutes = 0
        for i in hh:
            partialsum += i
            cumulativeH.append(partialsum)
            totRoutes = partialsum
        cum.set_xlim([0,25])
        cum.plot(bh[:len(cumulativeH)], 
                np.array(cumulativeH)/partialsum, color="green", 
                ls="steps--", linewidth=3)
                #label = "Cumulative hopCount")
        #cum.legend(loc="lower right")
        freq.plot(bh[:len(hh)], np.array(hh)/totRoutes, 
                label="length", color="green", drawstyle="steps", 
                linewidth=3)
        #freq.xaxis.grid(linestyle='--', linewidth=1, alpha=0.1)
        plt.savefig(routeFolder+"/routes."+C.imageExtension)
    

        plt.clf()
        sortedEtx = []
        for l in sorted(numHopMap):
            sortedEtx.append(numHopMap[l])
        #labels=(sorted(numHopMap))
        labels = []
        freqLabels = []
    
        weightDistributions = []
    
        bins = np.array(range(0,250,1))/float(10)
        odd = True
        for l in sorted(numHopMap):
            hw, bw = np.histogram(numHopMap[l], bins=bins)
            weightDistributions.append(hw)
            frac = 100*len(numHopMap[l])/float(totRoutes)
            if int(frac) > 0 :
                rFrac = str(int(100*len(numHopMap[l])/float(totRoutes)))
            else:
                rFrac = "."+str(int(frac*10))

            if odd:
                labels.append(str(l) + "\n" + str(rFrac) + "%")
                odd = False
            else:
                labels.append(str(l) + "\n\n" + str(rFrac) + "%")
                odd = True
            freqLabels.append(str(rFrac) + "%")
    
        ax1 = plt.subplot(111)
        #ax1.set_xticks(range(len(labels)), labels)
        ax1.set_xticklabels(labels, fontsize = 40)
        ax1.boxplot(sortedEtx)
        ax1.set_ylim([0,40])
        #ax2 = ax1.twiny()
        #ax2.set_xticks(range(0.5, len(freqLabels)))
        #ax2.set_xticklabels(freqLabels, fontsize = 14)
        #plt.text(0.5, 1.08, "Route weight Vs route length",
        #                 horizontalalignment='center',
        #                 fontsize=20,
        #                 transform = ax2.transAxes)
        plt.title("Route weight Vs route length")
        plt.savefig(routeFolder+"/boxplot."+C.imageExtension)
        plt.clf()
        
        i = 0
        for h in weightDistributions:
            i += 1
            #plt.plot(bins[:-1], h)
            plt.plot(bins[:-1], h, label="length "+str(i))
        plt.title("Distribution of the route weights")
        plt.xlabel("Route weight (ETX)")
        plt.ylabel("samples")
        plt.legend(loc="upper right", ncol=2)
        plt.savefig(routeFolder+"/etxWeights."+C.imageExtension)
        plt.clf()
    
        #s,b = np.histogram(etxStd, bins=100)
        #es = plt.subplot(1,1,1)
        #plt.title("ETX Standard deviation per route (density/cumulative)")
        #es.plot(b[:len(s)], s, label="ETX STD")
        #cum = es.twinx()
        #cumulative = []
        #partialsum = 0.0
        #for i in s:
        #    partialsum += i
        #    cumulative.append(partialsum)
        #cum.yaxis.tick_right()
        #cum.yaxis.set_label_position("right")
        #cum.set_ylim([0,1])
        #cum.plot(b[:len(cumulative)], 
        #        np.array(cumulative)/partialsum, 
        #        color = "green")
        ##cum.set_yscale("log")
        ##cum.set_xscale("log")
        #plt.savefig(routeFolder+"/00-ETX-STD.eps")
        #plt.clf()
    
    
        #sortedEtxPerCouple = []
        #for l in sorted(etxMap):
        #    sortedEtxPerCouple.append(etxMap[l])
        #labels=[int(x) for x in (sorted(etxMap))]
        #plt.xticks(range(len(labels)), labels)
        #plt.boxplot(sortedEtxPerCouple)
        #plt.title("Route weight max - route weight min Vs average length")
        #plt.savefig(routeFolder+"/00-ETX-box.eps")
        #plt.clf()
    
    def getDegreeDistribution(self, net):
        """ compute the distribution of the degree of nodes """
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

        retValue = dd2()
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
        ##thanks stackoverflow!
        fitfunc = lambda p, x: p[0] * x ** (p[1])
        errfunc = lambda p, x, y: (y - fitfunc(p, x))
        #
        out,success = optimize.leastsq(errfunc, 
                [1,-1],args=(x,y))
        fittedValue = []
        for v in x:
            fittedValue.append(out[0]*(v**out[1]))
        p = plt.subplot(1,1,1)
        p.plot(x, y, "ko", x, fittedValue, "k-", markersize=15)
        retValue["LINEAR"]["X"] = x
        retValue["LINEAR"]["Y"] = y
        retValue["LINEAR"]["FITTED"] = fittedValue
        #p.set_xscale("log")
        #p.set_yscale("log")
        #p.set_ylim(0,1)
        ##plt.ylim([0.0001,0])
        plt.title("Degree relative frequency (%)")
        plt.savefig(routeFolder+"/degree."+C.imageExtension)
        plt.clf()

        p = plt.subplot(1,1,1)
        p.plot(xNL, yNL, "ko", markersize=15)#, x, fittedValue)
        p.set_xscale("log")
        p.set_yscale("log")
        ##p.set_ylim(0,1)
        ##plt.ylim([0.0001,0])
        plt.title("Degree relative frequency, non-leaf subgraph")
        plt.savefig(routeFolder+"/NLdegree."+C.imageExtension)
        plt.clf()
        retValue["NLINEAR"]["X"] = x
        retValue["NLINEAR"]["Y"] = y
        retValue["NLINEAR"]["FITTED"] = fittedValue
        return retValue

    def diffVectors(self, v1,v2):
        """ compare two arrays """
        if len(v1) != len(v2):
            print >> sys.stderr, "Error, comparing two different arrays", v1, v2
            #sys.exit(1)
            #FIXME this happens sometimes
        diff = 0
        for item in v1:
            if item not in v2:
                diff += 1
        return diff
    
    
    def getCentralityMetrics(self, net):
        """ estimate centrality, this relies on community_network_analysis
        module """
        # FIXME cleanup this function
        bet = defaultdict(list) 
        betApproxCloseness = defaultdict(list) 
        betApproxDegree = defaultdict(list) 
        betApproxWeightedDegree = defaultdict(list) 
        singleNodeBetweenness = []
        singleNodeCloseness = []
        cl = defaultdict(list)
        betSol = defaultdict(list)
        clSol = defaultdict(list)
        counter = 0 # testing only, limit the number of graphs under analysis
        graphLimit = 10000
        firstSolution = set()
        solutionVariation = []
        for scanId in self.routeData:
            if counter >= graphLimit:
                print >> sys.stderr, "Exiting after", graphLimit, "tests"
                break
            counter += 1
            G = self.routeData[scanId]["Graph"]
            if len(G.nodes()) < 10:
                print >> sys.stderr, "ERROR, graph has only ", len(G.nodes()), "nodes!"
                continue
            # this will make a global graph of all the runs
            b = nx.betweenness_centrality(G)
            c = nx.closeness_centrality(G, distance=True, 
                    normalized=False)
            for n in G.nodes():
                if len(G[n]) > 1:
                    singleNodeBetweenness.append(b[n])
                    singleNodeCloseness.append(c[n])
            #singleNodeBetweenness += nx.betweenness_centrality(G).values()
            #singleNodeCloseness += nx.closeness_centrality(G, distance=True, 
            #        normalized=False).values()
    
            # this will call a multi-process routine that will compute the 
            # betweenness
            solBet, bestBet, solCl, bestCl, currCache = computeGroupMetrics(G, 
                    C.maxGroupSize, weighted=True, 
                    mode="greedy")
            if counter == 1:
                firstSolution = bestCl[C.maxGroupSize]
            b,c = groupMetricForOneGroup(G, firstSolution, currCache)
            solutionVariation.append(c - solCl[C.maxGroupSize])
            # this is used to approximate centrality with degree
            degreeDict = sorted(G.degree().items(), 
                    key = lambda x: x[1], reverse=True)
            highDegreeDict = [ i for i in reversed(degreeDict[:C.maxGroupSize]) ]
    
            unsortedNodeWeightedDegreeDict = []
            # this weights degree with the weight of the links
            for n in degreeDict:
                if n[1] == 1:
                    break
                nodeWeight = 0.0
                neighs = G[n[0]]
                for neigh in neighs:
                    nodeWeight += 1/G[n[0]][neigh]['weight'] 
                unsortedNodeWeightedDegreeDict.append((n[0], nodeWeight))
    
            nodeWeightedDegreeDict = sorted(unsortedNodeWeightedDegreeDict,
                    key = lambda x: x[1])[-C.maxGroupSize:]
    
            # Big Note: when the betweenness centrality is computed
            # using the degree to chose the group g, it may not be a monotone 
            # value anymore. This is due to the fact that when we compute the
            # centrality, we do not condier routes that start from a node in g.
            # A node x may have a large number of neighbors but all of them
            # with a bad quality, so no new routes pass throu x. but if we 
            # add x to g, we reduce the total number of routes R we use 
            # in the fraction to compute the average. As a result betweenness may
            # decrease. Consider the following network as an example of this:
            #
            #  1 -- 2 -- 3 -- 4
            #       |    X
            #       -- 5--
            # 
            # the link marked with X is very bad. At the first iteration g = [2]
            # do we have 14 routes passing through g and 2 not passing (3->4 and
            # 4->3). Then g = [2,5]. Since the routes starting from 5 are not 
            # counted anymore, the fraction is 10/(10+2) < 14/(14+2).
            # That's why we also use a weighted degree sorting
            #
    
            highClosenessDict = sorted(nx.closeness_centrality(G).items(), 
                    key = lambda x: x[1])[-C.maxGroupSize:]
    
            # compute the betweenness of some group of nodes
            for gSize in solBet:
                (b,c) = groupMetricForOneGroup(G, 
                        [x[0] for x in highClosenessDict][-gSize:], 
                        currCache)
                betApproxCloseness[gSize].append(b)
                (b,c) = groupMetricForOneGroup(G, 
                        [x[0] for x in highDegreeDict][-gSize:], 
                        currCache)
                betApproxDegree[gSize].append(b)
                (b,c) = groupMetricForOneGroup(G, 
                        [x[0] for x in nodeWeightedDegreeDict][-gSize:], 
                        currCache)
                betApproxWeightedDegree[gSize].append(b)
                bet[gSize].append(solBet[gSize])
                cl[gSize].append(solCl[gSize])
                betSol[gSize].append(bestBet[gSize])
                clSol[gSize].append(bestCl[gSize])
    
        avgBet = {}
        avgCl = {}
        avgBetApproxDegree = {}
        avgBetApproxWeightedDegree = {}
        avgBetApproxCloseness = {}
        diffBet = defaultdict(list)
        diffCl = defaultdict(list)
        
        # average the betweenness of multiple samples
        for gSize in bet:
            avgBet[gSize] = np.average(bet[gSize])
            avgCl[gSize] = np.average(cl[gSize])
            avgBetApproxDegree[gSize] = np.average(betApproxDegree[gSize])
            avgBetApproxWeightedDegree[gSize] = np.average(
                    betApproxWeightedDegree[gSize])
            avgBetApproxCloseness[gSize] = np.average(betApproxCloseness[gSize])
            prev = []
            curr = []
            for sol in betSol[gSize]:
                prev = [c for c in curr]
                curr = sol
                if prev != [] and curr != []:
                    diffBet[gSize].append(self.diffVectors(prev, curr))
            prev = []
            curr = []
            for sol in clSol[gSize]:
                prev = [c for c in curr]
                curr = sol
                if prev != [] and curr != []:
                    diffCl[gSize].append(self.diffVectors(prev, curr))
    
        centFolder = C.resultDir+"/CENTRALITY"
        retValue = defaultdict(dict)
        plt.clf()
        betG = plt.subplot(1,1,1)
        betG.set_ylim([0,1])
        betG.plot(avgBet.keys(), avgBet.values(), color="blue", 
                label="betweenness")
        retValue["BET"]["x"] = avgBet.keys()
        retValue["BET"]["y"] = avgBet.values()
        betG.plot(avgBet.keys(), avgBetApproxDegree.values(), color="red", 
                label="betweenness/degree-approx")
        retValue["BETD"]["x"] = avgBet.keys()
        retValue["BETD"]["y"] = 100*(np.array(avgBet.values()) - \
            np.array(avgBetApproxDegree.values()))/np.array(avgBet.values())

        betG.plot(avgBet.keys(), avgBetApproxCloseness.values(), color="gray", 
                label="betweenness/approx-closeness")
        betG.plot(avgBet.keys(), avgBetApproxWeightedDegree.values(), 
                color="green", label="betweenness/approx-weighted-degree")
        betG.legend(loc="lower right")
        betG.yaxis.grid(color='gray', linestyle='dashed')
        betG.set_axisbelow(True)
        plt.ylabel("Betweenness")
        plt.xlabel("Group size")
        plt.xticks(avgBet.keys())
        plt.title("Betweenness centrality - "+net)
        plt.savefig(centFolder+"/cent-betw-"+net+"."+C.imageExtension)
        plt.clf()
    
        clG = plt.subplot(1,1,1)
        clG.set_ylim([1, avgCl.values()[0]])
        clG.plot(avgCl.keys(), avgCl.values(), color="green", label="closeness")
        retValue["CLOS"]["x"] = avgCl.keys()
        retValue["CLOS"]["y"] = avgCl.values()
        retValue["CLOSV"]["x"] = range(len(solutionVariation))
        retValue["CLOSV"]["y"] = solutionVariation
        clG.legend(loc="lower right")
        plt.xlabel("Group size")
        plt.xticks(avgCl.keys())
        plt.ylabel("Closeness")
        plt.title("Closeness centrality - "+net)
        plt.savefig(centFolder+"/cent-cl-"+net+"."+C.imageExtension)
        plt.clf()
    
        for size in diffBet:
            plt.plot(range(len(diffBet[size])), diffBet[size], 
                    label="group size " + str(size))
        plt.title(net + ". Variations in the central sets per run, betweenness.")
        plt.xlabel("run id")
        plt.ylabel("differing elements from prev run")
        plt.legend(loc="lower right")
        plt.savefig(centFolder+"/cent-bet-variation-"+net+"."+C.imageExtension)
        plt.clf()
        for size in diffCl:
            plt.plot(range(len(diffCl[size])), diffCl[size],
                    label="group size " + str(size))
        plt.title(net + ". Variations in the central sets per run, closeness")
        plt.legend(loc="lower right")
        plt.savefig(centFolder+"/cent-cl-variation-"+net+"."+C.imageExtension)
        plt.clf()
        
        h,b = np.histogram(singleNodeBetweenness, bins=100) 
        totalSamples = sum(h)
        sb = plt.subplot(1,1,1)
        #sb.yaxis.tick_left()
        #sb.plot(b[1:], h, color="blue")
        plt.ylabel("# Samples")
        plt.xlabel("Betweenness")
    
        sumSamples = []
        partialSum = 0.0
        for v in h:
            partialSum += v
            sumSamples.append(partialSum)
        #sbSum=sb.twinx()
        sbSum = sb
        #sbSum.yaxis.tick_right()
        #sbSum.yaxis.set_label_position("right")
        sbSum.set_ylim([0,1])
        sbSum.plot(b[1:], (np.array(sumSamples)/totalSamples), 
                color="red")
        plt.title("Centrality histogram and normalized integral (all runs, non-leaf nodes)")
        plt.savefig(centFolder+"/betw-singleNode-"+net+"."+C.imageExtension)
        retValue["SINGLEB"]["x"] = b[1:]
        retValue["SINGLEB"]["y"] = \
                np.array(sumSamples)/totalSamples
    
        plt.clf()
        h,b = np.histogram([1/x for x in singleNodeCloseness], bins=100) 
        totalSamples = sum(h)
        sb = plt.subplot(1,1,1)
        sb.yaxis.tick_left()
        sb.plot(b[1:], h, color="blue")
        plt.title("Centrality histogram and normalized integral (all runs)")
        plt.xlabel("Closeness")
        plt.ylabel("# Samples")
    
        sumSamples = []
        partialSum = 0.0
        for v in h:
            partialSum += v
            sumSamples.append(partialSum)
        sbSum=sb.twinx()
        sbSum.yaxis.tick_right()
        sbSum.yaxis.set_label_position("right")
        sbSum.set_ylim([0,1])
        sbSum.plot(b[1:], (np.array(sumSamples)/totalSamples), 
                color="red")
        plt.savefig(centFolder+"/clos-singleNode-"+net+"."+C.imageExtension)
        plt.clf()

        retValue["SINGLEC"]["x"] = b[1:]
        retValue["SINGLEC"]["y"] = \
                np.array(sumSamples)/totalSamples
        return retValue



    def getMPRSets(self, net, mprMode):
        """ estimate MPR sets, and other parameters related to that """

        routeFolder = C.resultDir+net+"/ROUTES/"
        counter = 70000 # testing only, limit the number of graphs under analysis
        mpr = []
        IPUDPHEaderSize = 20 + 8 # bytes
        OLSRMsgHeaderSize = 16 # bytes
        TCMsgHeaderSize = 4 # bytes
        SelectorFieldSize = 4 + 4# bytes (IP + quality)

        TCPeriod = 5.0 # seconds
        signallingTraffic = []
        tcMessages = defaultdict(list)
        worstCaseTraffic = []
        mainCSize = defaultdict(list)
        relativeMainCSize = defaultdict(list)
        selectorSetArray = []
        averageRobustness = defaultdict(list)
        for scanId in self.routeData:
            selectorSet = defaultdict(set)
            if counter <= 0:
                return
            counter -= 1
            G = self.routeData[scanId]["Graph"]
            if len(G.nodes()) < 10:
                print >> sys.stderr, "ERROR, graph has only ", len(G.nodes()),\
                        "nodes!"
                continue
            mprSets = solveMPRProblem(G, mode=mprMode)
            G = self.routeData[scanId]["Graph"]
            weightStats = routeComparison(G, mprSets)
            for couple, route in weightStats.items():
                self.weightStats[couple] += route

            purgedG = purgeNonMPRLinks(G, mprSets, metric="weight")
            globalMPRSet = set()
            for node in mprSets:
                for mSolution in mprSets[node]:
                    # this returns an array of frozensets
                    # i want only the first one (see break)
                    for m in mSolution:
                        selectorSet[m].add(node)
                    break
                globalMPRSet |= mprSets[node].pop()
            mpr.append(globalMPRSet)

            TCtraffic = 0
            for m in selectorSet:
                TCtraffic += (IPUDPHEaderSize + OLSRMsgHeaderSize + \
                        TCMsgHeaderSize +\
                        len(selectorSet[m])*SelectorFieldSize)
            signallingTraffic.append(TCtraffic*len(selectorSet))
            #signallingTraffic.append(TCtraffic)
            tcMessages["MPR"].append(
                    1.0*len(selectorSet)*(len(selectorSet)-1)/
                    len(G.edges()))
            tcMessages["WC"].append(
                    1.0*len(G)*(len(G)-1)/len(G.edges()))


            TCtraffic = 0
            for node in G.nodes():
                TCtraffic += (IPUDPHEaderSize + OLSRMsgHeaderSize +\
                        TCMsgHeaderSize + len(G[node])*SelectorFieldSize)
            worstCaseTraffic.append(TCtraffic*len(G.nodes()))

            r = computeRobustness(purgedG, tests=30)[0]
            for k,v in r.items():
                percent = int(100*float(k)/len(globalMPRSet))
                averageRobustness[percent].append(v)

            #for k in m:
            #    mainCSize[k].append(m[k])
            #    relativeMainCSize[float(k)/len(globalMPRSet)].append(m[k])
            selectorSetArray.append(selectorSet)

        retValue = defaultdict(dict)

        # FIXME make this threshold dynamic. -r is needed
        tailCut = 5
        retValue["ROBUSTNESS"]["x"] = sorted(
                averageRobustness.keys()[:-tailCut])
        retValue["ROBUSTNESS"]["y"] = [
                np.average(averageRobustness[key]) for key in \
                retValue["ROBUSTNESS"]["x"]]

        plt.plot(retValue["ROBUSTNESS"]["x"], 
                retValue["ROBUSTNESS"]["y"])
        plt.title("Average robustness: "+self.net)
        plt.xlabel("Number of failed links")
        plt.savefig(routeFolder+"/robustness-"+mprMode+"."+C.imageExtension)
        plt.clf()

        if self.net == "FFGraz":
            retValue["MPR"]["x"] = range(0,len(mpr)*2, 2)
        else:
            retValue["MPR"]["x"] = range(len(mpr))
        retValue["MPR"]["y"] = [len(x) for x in mpr]
        plt.plot(retValue["MPR"]["x"], retValue["MPR"]["y"])
        plt.title("Global MPR set size, mode=\""+mprMode+"\","+net)
        plt.ylim([0,150])
        plt.xlabel("snapshots")
        plt.ylabel("global MPR set size")
        plt.savefig(routeFolder+"/mpr-set-size-"+net+"-"+mprMode+"."+C.imageExtension)
        plt.clf()

        mprSet = [] 
        mprDiff = []
        for ss in selectorSetArray:
            prevMPRSet = copy(mprSet)
            mprSet = ss
            currDiff = 0
            # for each mpr compare its selector set with the
            # selector set from the previous run. The metric
            # adds 1 every time the selector set of an MPR changes
            # from one snapshot to another, so that the routing
            # tables must be recomputed
            for mpr in ss.keys():
                if mpr in prevMPRSet:
                    prev = prevMPRSet[mpr]
                    curr = mprSet[mpr]
                    #intersection = prev.intersection(curr)
                    #diff = len(prev) - len(intersection) + len(curr) - len(intersection)
                    #currDiff += diff
                    if prev != curr:
                        currDiff += 1
                else:
                    currDiff += 1
            mprDiff.append(float(currDiff))
        retValue["MPRDIFF"]["x"] = range(len(mprDiff)-1)
        retValue["MPRDIFF"]["y"] = mprDiff[1:]


        #mprDiff = []
        #prev = set()
        #curr = set()
        #for mprs in mpr:
        #    prev = set([c for c in curr])
        #    curr = mprs
        #    if prev != set() and curr != set():
        #        intersection = prev.intersection(curr)
        #        diff = len(prev) - len(intersection) + len(curr) - len(intersection)
        #        mprDiff.append(diff)
    
        #plt.plot(range(len(mprDiff)), mprDiff)
        #plt.title("Global MPR set diff from one sample to the next one,"+\
        #        "mode=\""+mprMode+"\","+net)
        #plt.xlabel("snapshots")
        #plt.ylabel("elements that differ")
        #plt.savefig(routeFolder+"/mpr-diff-"+net+"-"+mprMode+"."+C.imageExtension)
        #plt.clf()
        bestW = []
        mprW = []
        for v in self.weightStats.values():
            vz = zip(*v)
            bestW.append(np.average(vz[0]))
            mprW.append(np.average(vz[1]))

        sbestW = []
        smprW = []
        diffSamples = 0
        counter = 0
        for i in sorted(bestW):
            sbestW.append(i)
            smprW.append(mprW[bestW.index(i)])
            if sbestW[counter] != smprW[counter]:
                diffSamples += 1
            counter += 1
        plt.plot(range(len(sbestW)), smprW,
                label="approx-weight", linewidth=2, color="yellow")
        plt.plot(range(len(sbestW)), sbestW,
                label="optimal-weight", linewidth=5, color="red", alpha=0.5)
        #plt.axes().set_aspect(2)
        #plt.plot(range(len(smprW)), smprW, label="approx-weight")
        #plt.plot(range(len(sbestW)), sbestW, label="optimal-weight")
        #plt.plot(range(len(smprW)), smprW, label="approx-weight")
        plt.xlabel("route")
        plt.ylabel("Route weight")
        plt.title("Comparison of route weight")
        plt.legend()
        plt.savefig(routeFolder+"/weight-approx-"+net+"-"+mprMode+"."+\
                C.imageExtension)
        plt.clf()
        self.weightStats = dd1()

        retValue["STRAFFIC"] = np.average(signallingTraffic)
        retValue["WCTRAFFIC"] = np.average(worstCaseTraffic)
        retValue["MPRTC"] = np.average(tcMessages["MPR"])
        retValue["WCTC"] = np.average(tcMessages["WC"])
        return retValue

def extractDataSeries(retValues):
    """ This function is called once the single processes have exited. It
    collects the data from each process and produces comparison results """

    etx = dataPlot(C)
    link = dataPlot(C)
    degree = dataPlot(C)
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
        routeFolder = C.resultDir+n+"/ROUTES/"
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
            #link.x.append(v["link"]["x"])
            #link.x.append(v["link"]["x"])
            link.y.append((v["link"]["avg"], n))
            #link.y.append((v["link"]["sup"], ""))
            #link.y.append((v["link"]["inf"], ""))
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

        if "degree" in v:
            degree.x.append(v["degree"]["LINEAR"]["X"])
            degree.x.append(v["degree"]["LINEAR"]["X"])
            degree.y.append((v["degree"]["LINEAR"]["Y"], ""))
            degree.y.append((v["degree"]["LINEAR"]["FITTED"],""))
            degree.outFile = routeFolder + "degreeDistribution"

        if "MPRRFC" in v:
            mprRFC.x.append(v["MPRRFC"]["MPR"]["x"])
            mprRFC.y.append((v["MPRRFC"]["MPR"]["y"], n))
            mprRFC.title = "Size of the global MPR set (RFC)"
            mprRFC.outFile = comparisonFolder+"mpr-rfc"
            mprRFC.xAxisLabel = "snapshot"
            link.legendPosition = "lower right"

        if "MPRLQ" in v:
            mprLq.x.append(v["MPRLQ"]["MPR"]["x"])
            mprLq.y.append((v["MPRLQ"]["MPR"]["y"], n))
            mprLq.title = "Size of the global MPR set (lq)"
            mprLq.outFile = comparisonFolder+"mpr-lq"
            mprLq.xAxisLabel = "snapshot"
            link.legendPosition = "lower right"

        if "MPRLQ" in v and "MPRRFC" in v:
            mprRobustness.x.append(v["MPRRFC"]["ROBUSTNESS"]["x"])
            mprRobustness.y.append((v["MPRRFC"]["ROBUSTNESS"]["y"], n+"-RFC"))
            mprRobustness.x.append(v["MPRLQ"]["ROBUSTNESS"]["x"])
            mprRobustness.y.append((v["MPRLQ"]["ROBUSTNESS"]["y"], n+"-lq"))
            mprRobustness.title = "Robustness metric of the MPR sub-graph"
            mprRobustness.outFile = comparisonFolder+"mprrobustness"
            mprRobustness.xAxisLabel = "100* Failed links/mpr number"
            mprRobustness.yAxisLabel = "Robustness"
            mprRobustness.legendPosition = "lower left"

            #if "MPRSIGRFC" in v["MPRRFC"] and "MPRSIGLQ" in v["MPRLQ"]:
            MPRSigHistogram["RFC"].append(v["MPRRFC"]["MPRTC"])
            MPRSigHistogram["LQ"].append(v["MPRLQ"]["MPRTC"])
            MPRSigHistogram["WC"].append(v["MPRLQ"]["WCTC"])
            MPRSigHistogram["label"].append(n)


        if "CENTRALITY" in v:
            singleNodeCloseness.x.append(v["CENTRALITY"]["SINGLEC"]["x"])
            singleNodeCloseness.y.append((v["CENTRALITY"]["SINGLEC"]["y"],n))
            singleNodeCloseness.title = "Closeness ECDF, non-leaf nodes"
            singleNodeCloseness.outFile = comparisonFolder + "sgCloseness"
            singleNodeCloseness.xAxisLabel = "Closeness"
            singleNodeCloseness.yAxisLabel = ""
            singleNodeCloseness.legendPosition = "lower right"

            singleNodeBetweenness.x.append(v["CENTRALITY"]["SINGLEB"]["x"])
            singleNodeBetweenness.y.append((v["CENTRALITY"]["SINGLEB"]["y"],n))
            singleNodeBetweenness.title = "Betweenness ECDF, non-leaf nodes"
            singleNodeBetweenness.outFile = comparisonFolder + "sgBetweenness"
            singleNodeBetweenness.xAxisLabel = "Betweenness"
            singleNodeBetweenness.yAxisLabel = ""
            singleNodeBetweenness.legendPosition = "lower right"
            closeness.y.append((v["CENTRALITY"]["CLOS"]["y"], n))
            closeness.x.append(v["CENTRALITY"]["CLOS"]["x"])
            closeness.title = "Group closeness centrality"
            closeness.outFile = comparisonFolder+"closeness"
            closeness.xAxisLabel = "Group Size"
            closeness.yAxisLabel = "Group Closeness"
            closeness.legendPosition = "lower left"

            closenessV.x.append(range(len(v["CENTRALITY"]["CLOSV"]["x"])))
            closenessV.y.append((v["CENTRALITY"]["CLOSV"]["y"], n))
            closenessV.title = "Group closeness centrality variation"
            closenessV.outFile = comparisonFolder+"closeness-variation"
            closenessV.xAxisLabel = "snapshot"
            closenessV.yAxisLabel = "Group Closeness (5 nodes)"
            closenessV.legendPosition = "upper left"

            betweenness.x.append(v["CENTRALITY"]["BET"]["x"])
            betweenness.y.append((v["CENTRALITY"]["BET"]["y"], n))
            betweenness.title = "Group betweenness centrality"
            betweenness.outFile = comparisonFolder+"betweenness"
            betweenness.xAxisLabel = "Group Size"
            betweenness.yAxisLabel = "Group Betweenness"
            betweenness.legendPosition = "lower right"

            betweennessD.x.append(v["CENTRALITY"]["BETD"]["x"])
            betweennessD.y.append((v["CENTRALITY"]["BETD"]["y"], n))
            betweennessD.title = "Group betweenness (highest degree)"
            betweennessD.outFile = comparisonFolder+"betweenness-estimation"
            betweennessD.xAxisLabel = "Group Size"
            betweennessD.yAxisLabel = "difference from the best group Betweenness (%)"
            betweennessD.legendPosition = "upper right"

        if "ROBUSTNESS" in v:
            robustness.x.append(v["ROBUSTNESS"]["RB"]["x"])
            robustness.y.append((v["ROBUSTNESS"]["RB"]["y"],n))
            robustness.x.append(v["ROBUSTNESS"]["CRB"]["x"])
            robustness.y.append((v["ROBUSTNESS"]["CRB"]["y"],n+" core"))
            robustness.title = "Robustness metrics"
            robustness.xAxisLabel = "Failed links (%)"
            robustness.legendPosition = "lower left"
            robustness.outFile = comparisonFolder+"graphrobustness"

    etx.plotData(bw="PRINTABLE")
    etx.saveData()
    link.plotData(bw="PRINTABLE")
    link.saveData()
    degree.saveData()
    mprLq.plotData(bw="PRINTABLE")
    mprRFC.plotData(bw="PRINTABLE")
    mprRobustness.plotData(bw="PRINTABLE")
    closeness.plotData(bw="PRINTABLE")
    betweenness.plotData(bw="PRINTABLE")
    betweennessD.plotData()
    closenessV.plotData()
    robustness.plotData(bw="PRINTABLE")
    singleNodeCloseness.plotData()
    singleNodeBetweenness.plotData()


    # some of the graph must be plotted with special configurations so that they
    # are readable 

    if len(MPRSigHistogram) != 0:
        ax = plt.subplot(111)
        #ax = fig.subplot(111)
        shift = 0.3
        width = 0.3
        x = np.array(range(1,len(MPRSigHistogram["RFC"])+1))
        ax.bar(x, MPRSigHistogram["RFC"], width, color = "b", label="RFC")
        ax.bar(shift + x, MPRSigHistogram["LQ"], width, color = "r", label="lq")
        ax.bar(2*shift + x, MPRSigHistogram["WC"], width, color = "y", 
                label="LSR")
        ax.set_xticks(x+shift)
        lb = ax.set_xticklabels(MPRSigHistogram["label"])
        plt.setp(lb,rotation=45)
        plt.legend(loc="upper left")
        plt.title("TC messages per link per TC emission interval")
        plt.savefig(comparisonFolder+"signallingHist."+C.imageExtension)
        plt.clf()

    if robustness.title !=  "":
        # robustness graphs need to be printed with custom options.
        plt.clf()
        colors = ["blue", "red", "#eec223"]
        thickNess = [1,3,5]
        for i in range(len(robustness.x)):
            if "core" in robustness.y[i][1]:
                plt.plot(robustness.x[i], robustness.y[i][0], 
                        label=robustness.y[i][1], 
                        linestyle="--", color=colors[i/2],
                        linewidth=thickNess[i/2])
            else:
                plt.plot(robustness.x[i], robustness.y[i][0], 
                        label=robustness.y[i][1], color=colors[i/2],
                        linewidth=thickNess[i/2])

        plt.legend(loc="lower left", fancybox=True, 
                 shadow=True, numpoints=1)
        plt.title(robustness.title)
        plt.xlabel(robustness.xAxisLabel)
        plt.ylabel(robustness.yAxisLabel)
        plt.savefig("/tmp/rob.eps")

    if mprRobustness.title !=  "":
        # robustness graphs need to be printed with custom options.
        plt.clf()
        colors = ["blue", "red", "#eec223"]
        thickNess = [1,3,5]
        for i in range(len(mprRobustness.x)):
            if "RFC" in mprRobustness.y[i][1]:
                plt.plot(mprRobustness.x[i], mprRobustness.y[i][0], 
                        label=mprRobustness.y[i][1], 
                        linestyle="--", color=colors[i/2],
                        linewidth=thickNess[i/2])
            else:
                plt.plot(mprRobustness.x[i], mprRobustness.y[i][0], 
                        label=mprRobustness.y[i][1], color=colors[i/2],
                        linewidth=thickNess[i/2])

        plt.legend(loc="lower left", fancybox=True, 
                 shadow=True, numpoints=1)
        plt.title(robustness.title)
        plt.xlabel(robustness.xAxisLabel)
        plt.ylabel(robustness.yAxisLabel)
        plt.savefig("/tmp/MPRrob.eps")



class configuration:
    """ A configuration class to store command line and other parameters """
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
        self.imageExtension = "eps"
        self.nameCompression = ""
        self.namesDictionaryFileName = ""
        self.decryptKey = ""
        self.myCrypto = None
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
        print "-n name compression JSON file"

def createFolder(folder):
    """ check if the folder exists and create it """
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
        opts, args = getopt.getopt(sys.argv[1:], "d:f:r:s:pS:vn:k:")
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
        if option == "-k":
            C.decryptKey = v
            continue
        if option == "-n":
            C.namesDictionaryFileName = v
            continue
    if not C.checkCorrectness():
        sys.exit(1)


    rcParams.update({'font.size': 40})
    startTime =  datetime.now()
    print "loading", datetime.now()
    print C.loadDb
    if C.loadDb != "":
        engine = create_engine(C.loadDb)
        sessionFactory = sessionmaker(bind=engine, autocommit=True)
        localSession = scoped_session(sessionFactory)
        C.myCrypto = myCrypto(C.decryptKey)
        getDataSummary(localSession, data)

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
	    	data.dataSummary[net],
            data.namesDictionary))
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


