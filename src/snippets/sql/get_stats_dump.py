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

try:
    from groupMetrics import computeGroupMetrics, groupMetricForOneGroup
    from mpr import solveMPRProblem, purgeNonMPRLinks
except ImportError:
    print >> sys.stderr, "ERROR: You must link into this folder mpr.py, groupMetrics.py and ",\
    "miscLibs.py from the community network analyser tool"
    sys.exit()

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
        self.linkData = dd3()
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
        self.linkData = d.linkData
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
                logString += str("ID").ljust(3)
                for label in data.dataSummary[net][sid]:
                    logString += label[0].ljust(label[1]) + " "
                logString += "\n"
                break
            # print the data
            for sid in sorted(data.dataSummary[net]):
                logString += str(sid).ljust(3)
                for key, value in data.dataSummary[net][sid].items():
                    logString += str(value).ljust(key[1]) + " "
                logString += "\n"
        logString += "\n\nETX threshold:" + str(self.etxThreshold)

        return logString
             
            
         

def getDataSummary(ls, data):
    scanQuery = "SELECT * from scan"
    QUERY="""select snode.Id AS sid, dnode.Id AS did, etx.etx_value AS etxv from \
           link, scan, node as snode, node as dnode, etx \
           WHERE link.scan_Id = scan.Id AND snode.Id = link.from_node_Id \
           AND dnode.Id = link.to_node_Id AND etx.link_Id = link.Id \
           AND dnode.scan_Id = scan.Id AND snode.scan_Id = scan.Id AND scan.Id= %d"""

    try:
        q = ls.query("Id", "time", "scan_type", "network").from_statement(
                scanQuery)
        if len(q.all()) == 0:
            raise
    except:
        print "something went wrong with opening the db"
        sys.exit(1)

    numScan = len(q.all())
    scanCounter = 0
    data.etxThreshold = C.etxThreshold
    for [scanId, scanTime, scanType, scanNetwork] in q:
        data.scanTree[scanNetwork][scanType].append([scanId, scanTime])
    
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
            q = ls.query("sid", "did", "etxv").\
                    from_statement(queryString)
            dirtyG = nx.Graph()
            for s,d,e in q:
                if e < C.etxThreshold:
                    dirtyG.add_edge(s,d, weight=float(e))
                    data.linkData[net][s][d].append(e)

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


            data.dataSummary[net][scanId[0]][("time", 30)] = scanId[1]
            data.dataSummary[net][scanId[0]][("numNodes",9)] = len(dirtyG)
            data.dataSummary[net][scanId[0]][("numEdges",9)] = len(etxV)
            data.dataSummary[net][scanId[0]][("largestComponent",16)] = \
                    componentSize
            data.dataSummary[net][scanId[0]][("etxAvg",8)] = \
                    str(np.average(etxV))[0:5]
            scanCounter += 1
            if int((100000 * 1.0*scanCounter / numScan)) % 10000 == 0:
                print int((100 * 1.0*scanCounter / numScan)),"% complete"


class dataParser():
    net = "" 
    q = None


    def __init__(self, netName, queue):
        self.net = netName
        self.q = queue
    def run(self):
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
        for scanId in data.rawData[self.net]:
            netEtx += data.rawData[self.net][scanId]
        self.getRouteDistributions(self.net)
        self.getDegreeDistribution(self.net)
        bins = np.array(range(1,1000))/100.0
        retValue["etx"] = self.getETXDistribution(netEtx, bins)
        retValue["link"] = self.getLinkDistributions(self.net)
        retValue["CENTRALITY"] = self.getCentralityMetrics(self.net)
        mprRFC = self.getMPRSets(self.net, "RFC")
        retValue["MPRRFC"] = mprRFC
        mprLq = self.getMPRSets(self.net, "lq")
        retValue["MPRlq"] = mprLq
        #print "XXX", wc, lqTraffic, rfcTraffic
        q.put(retValue)

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
        linkFolder = C.resultDir+net+"/LINKS/"
        try:
            os.mkdir(linkFolder)
        except:
            pass
        etxRanking = []
        for s in data.linkData[net]:
            for d,vArray in data.linkData[net][s].items():
                # considering only the links that appear in 
                # at least 30% of the samples
                if len(vArray) < 0.2*len(data.dataSummary[net]):
                    continue
                cleanList = [ e for e in vArray if e < 20]
                avgEtx = np.average(cleanList)
                avgStd = np.std(cleanList)
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
        for scan in data.routeData[net]:
            D = data.routeData[net][scan]['data']
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
        maxETX = int(max(etxList))
        hh,bh = np.histogram(numHops, bins=maxHops)
        he,be = np.histogram(etxList, bins=maxETX)
        plt.xlabel("ETX weight/hop count")
        freq = plt.subplot(1,1,1)
        ep = freq.plot(be[:len(he)], he, label="ETX")
        hp = freq.plot(bh[:len(hh)], hh, label="numHops")
        cum = freq.twinx()
        cum.yaxis.tick_right()
        cum.yaxis.set_label_position("right")
        cum.set_ylim([0,1])
        #plt.legend([ep,hp], ["etx", "hopCount"])
        freq.legend(loc="center right")
        cumulativeH = []
        cumulativeE = []
        partialsum = 0.0
        plt.title("Path lenght/weight distribution")
        for i in he:
            partialsum += i
            cumulativeE.append(partialsum)
    
        cum.plot(be[:len(cumulativeE)], 
                np.array(cumulativeE)/partialsum, "--")
                #label = "Cumulative ETX")
        partialsum = 0.0
        totRoutes = 0
        for i in hh:
            partialsum += i
            cumulativeH.append(partialsum)
            totRoutes = partialsum
        cum.set_xlim([0,25])
        cum.plot(bh[:len(cumulativeH)], 
                np.array(cumulativeH)/partialsum, "--")
                #label = "Cumulative hopCount")
        cum.legend(loc="lower right")
        plt.savefig(routeFolder+"/routes."+C.imageExtension)
    
        plt.clf()
        sortedEtx = []
        for l in sorted(numHopMap):
            sortedEtx.append(numHopMap[l])
        #labels=(sorted(numHopMap))
        labels = []
    
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
    
        plt.xticks(range(len(labels)), labels, fontsize=14)
        plt.boxplot(sortedEtx)
        plt.title("route weight Vs route lenght")
        plt.savefig(routeFolder+"/boxplot."+C.imageExtension)
        plt.clf()
        
        i = 0
        for h in weightDistributions:
            i += 1
            plt.plot(bins[:-1], h)
            #plt.plot(bins[:-1], h, label="route length "+str(i))
        plt.title("Distribution of the route weights")
        plt.xlabel("route weight (ETX)")
        plt.ylabel("samples")
        plt.legend(loc="upper right")
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
        #plt.title("Route weight max - route weight min Vs average lenght")
        #plt.savefig(routeFolder+"/00-ETX-box.eps")
        #plt.clf()
    
    def getDegreeDistribution(self, net):
        routeFolder = C.resultDir+net+"/ROUTES"
        degreeDistribution = defaultdict(float)
        samples = 0.0
        for scanId in data.routeData[net]:
            graph = data.routeData[net][scanId]["Graph"]
            for node in graph:
                degreeDistribution[graph.degree(node)] += 1
                samples += 1
        for d in degreeDistribution:
            degreeDistribution[d] /= samples
        x = degreeDistribution.keys()
        y = degreeDistribution.values()
    
        #thanks stackoverflow!
        fitfunc = lambda p, x: p[0] * x ** (p[1])
        errfunc = lambda p, x, y: (y - fitfunc(p, x))
        
        out,success = optimize.leastsq(errfunc, 
                [1,-1],args=(x,y))
        fittedValue = []
        for v in x:
            fittedValue.append(out[0]*(v**out[1]))
        p = plt.subplot(1,1,1)
        p.plot(x, degreeDistribution.values(), "ro")#, x, fittedValue)
        p.set_xscale("log")
        p.set_yscale("log")
        ##p.set_ylim(0,1)
        ##plt.ylim([0.0001,0])
        plt.title("degree frequency graph (m=%s)"%str(out[1])[0:6])
        plt.savefig(routeFolder+"/degree."+C.imageExtension)
        plt.clf()
    
    def diffVectors(self, v1,v2):
        if len(v1) != len(v2):
            print >> sys.stderr, "Error, comparing two different arrays", v1, v2
            sys.exit(1)
        diff = 0
        for item in v1:
            if item not in v2:
                diff += 1
        return diff
    
    
    def getCentralityMetrics(self, net):
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
        graphLimit = 10
        firstSolution = set()
        solutionVariation = []
        for scanId in data.routeData[net]:
            if counter >= graphLimit:
                print >> sys.stderr, "Exiting after", graphLimit, "tests"
                break
            counter += 1
            G = data.routeData[net][scanId]["Graph"]
            if len(G.nodes()) < 10:
                print >> sys.stderr, "ERROR, graph has only ", len(G.nodes()), "nodes!"
                continue
            # this will make a global graph of all the runs
            singleNodeBetweenness += nx.betweenness_centrality(G).values()
            singleNodeCloseness += nx.closeness_centrality(G, distance=True, 
                    normalized=False).values()
    
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
        retValue["CLOSV"] = solutionVariation
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
        sb.yaxis.tick_left()
        sb.plot(b[1:], h, color="blue")
        plt.ylabel("# Samples")
        plt.xlabel("Betweenness")
    
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
        plt.title("Centrality histogram and normalized integral (all runs)")
        plt.savefig(centFolder+"/betw-singleNode-"+net+"."+C.imageExtension)
    
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
        return retValue
    
    
    
    def getMPRSets(self, net, mprMode):
        routeFolder = C.resultDir+net+"/ROUTES/"
        counter = 700 # testing only, limit the number of graphs under analysis
        mpr = []
        IPUDPHEaderSize = 20 + 8 # bytes
        OLSRMsgHeaderSize = 16 # bytes
        TCMsgHeaderSize = 4 # bytes
        SelectorFieldSize = 4 + 4# bytes (IP + quality)

        TCPeriod = 5.0 # seconds
        signallingTraffic = []
        worstCaseTraffic = []
        mainCSize = defaultdict(list)
        for scanId in data.routeData[net]:
            selectorSet = defaultdict(set)
            if counter <= 0:
                return
            counter -= 1
            G = data.routeData[net][scanId]["Graph"]
            if len(G.nodes()) < 10:
                print >> sys.stderr, "ERROR, graph has only ", len(G.nodes()),\
                        "nodes!"
                continue
            mprSets = solveMPRProblem(G, mode=mprMode)
            purgedG = purgeNonMPRLinks(G, mprSets)
            globalMPRSet = set()
            for node in mprSets:
                for m in mprSets[node]:
                    selectorSet[m].add(node)
                globalMPRSet |= mprSets[node].pop()
            mpr.append(globalMPRSet)

            TCtraffic = 0
            for m in selectorSet:
                TCtraffic += (IPUDPHEaderSize + OLSRMsgHeaderSize + TCMsgHeaderSize +\
                        len(selectorSet[m])*SelectorFieldSize)/TCPeriod
            signallingTraffic.append(TCtraffic*len(selectorSet)/len(G.edges()))
            #signallingTraffic.append(TCtraffic)

            TCtraffic = 0
            for node in G.nodes():
                TCtraffic += (IPUDPHEaderSize + OLSRMsgHeaderSize + TCMsgHeaderSize +\
                        len(G[node])*SelectorFieldSize)/TCPeriod
            worstCaseTraffic.append(TCtraffic*len(G.nodes())/len(G.edges()))

            m, nm = self.computeRobustness(purgedG, tests=10)

            relativeMainCSize = defaultdict(list)
            for k in m:
                mainCSize[k].append(m[k])
                relativeMainCSize[float(k)/len(globalMPRSet)].append(m[k])
        retValue = defaultdict(dict)

        # FIXME make this threshold dynamic. -r is needed
        tailCut = 5
        retValue["ROBUSTNESS"]["x"] = relativeMainCSize.keys()[:-tailCut]
        retValue["ROBUSTNESS"]["y"] = [np.average(relativeMainCSize[key]) for key in \
                relativeMainCSize.keys()[:-tailCut]]

        plt.plot(retValue["ROBUSTNESS"]["x"], retValue["ROBUSTNESS"]["y"])
        plt.title("Average robustness: "+self.net)
        plt.xlabel("Number of failed links")
        plt.savefig(routeFolder+"/robustness-"+mprMode+"."+C.imageExtension)
        plt.clf()

        retValue["MPR"]["x"] = range(len(mpr))
        retValue["MPR"]["y"] = [len(x) for x in mpr]
        plt.plot(retValue["MPR"]["x"], retValue["MPR"]["y"])
        plt.title("Global MPR set size, mode=\""+mprMode+"\","+net)
        plt.ylim([0,150])
        plt.xlabel("snapshots")
        plt.ylabel("global MPR set size")
        plt.savefig(routeFolder+"/mpr-set-size-"+net+"-"+mprMode+"."+C.imageExtension)
        plt.clf()
        mprDiff = []
        prev = set()
        curr = set()
        for mprs in mpr:
            prev = set([c for c in curr])
            curr = mprs
            if prev != set() and curr != set():
                intersection = prev.intersection(curr)
                diff = len(prev) - len(intersection) + len(curr) - len(intersection)
                mprDiff.append(diff)
    
        plt.plot(range(len(mprDiff)), mprDiff)
        plt.title("Global MPR set diff from one sample to the next one,"+\
                "mode=\""+mprMode+"\","+net)
        plt.xlabel("snapshots")
        plt.ylabel("elements that differ")
        plt.savefig(routeFolder+"/mpr-diff-"+net+"-"+mprMode+"."+C.imageExtension)
        plt.clf()

        retValue["STRAFFIC"] = 8*np.average(signallingTraffic)
        retValue["WTRAFFIC"] = 8*np.average(signallingTraffic)
        return retValue

    def computeRobustness(self, graph, tests=100):

        links = []
        weights = []
        for l in graph.edges(data=True):
            links.append((l[0], l[1])) 
            #weights.append(float(l[2]["weight"]))
        #totWeight = sum(weights)
        #normalizedWeight = [l/totWeight for l in weights]
        normalizedWeight = [1.0/len(links) for l in links]
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
                mainNonCSize[k].append(
                        np.average([len(r) for r in compList[1:]])/nlen)
        mainCSizeAvg = {}
        for k, tests in mainCSize.items():
            mainCSizeAvg[k] = np.average(tests)
        return mainCSizeAvg, mainNonCSize
        #    [np.average(mainCSize[k]) for k in sorted(mainCSize.keys())], label="Main C.")
        #plt.plot(np.array(sorted(mainCSize.keys()))/elen,
        #    [np.average(mainCSize[k]) for k in sorted(mainCSize.keys())], label="Main C.")
        ##FIXME add confidence interval
        ##FIXME do the same for nodes

        ##plt.plot(sorted(mainCSize.keys()),
        ##    [mainNonCSize[k] for k in sorted(mainNonCSize.keys())],
        ##    label = "Average minor C.")

        #plt.savefig("/tmp/robustness-"+self.net+"."+C.imageExtension)
        #plt.clf()
        

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

    def plotData(self):
        dataDimension = 0
        ax = plt.subplot(111)
        for y in self.y:
            l = self.y[dataDimension][1]
            v = self.y[dataDimension][0]
            if l != "": 
                ax.plot(self.x[dataDimension],
                    v, label=l)
            else :
                ax.plot(self.x[dataDimension], v)
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
                prop={'size':12})
        else: 
            plt.legend(loc=self.legendPosition, fancybox=True, 
                shadow=True)
        plt.savefig(self.outFile+self.fileType)
        plt.clf()


def extractDataSeries(parsers):

    etx = dataPlot(C)
    link = dataPlot(C)
    mprLq = dataPlot(C)
    mprRFC = dataPlot(C)
    betweenness = dataPlot(C)
    betweennessD = dataPlot(C)
    closeness = dataPlot(C)
    mprRobustness = dataPlot(C)
    closenessV = dataPlot(C)
    retValues = defaultdict(dict)
    for (n,p,v) in parsers:
        retValues[n] = v.get()

    for n,v in retValues.items():
        etx.y.append((v["etx"]["y"], n) )
        etx.x.append(v["etx"]["x"])
        etx.title = "ETX ECDF"
        etx.outFile = "/tmp/etx"
        etx.xAxisLabel = "ETX"
        etx.yAxisLabel = ""

        link.x.append(v["link"]["x"])
        link.x.append(v["link"]["x"])
        link.x.append(v["link"]["x"])
        link.y.append((v["link"]["avg"], n))
        link.y.append((v["link"]["sup"], ""))
        link.y.append((v["link"]["inf"], ""))
        link.outFile = "/tmp/link"
        link.title  = "average ETX per link +/- stddev"
        link.xAxisLabel = "link"
        link.yAxisLabel = "ETX"
        link.legendPosition = "upper left"

        mprRFC.x.append(v["MPRRFC"]["MPR"]["x"])
        mprRFC.y.append((v["MPRRFC"]["MPR"]["y"], n))
        mprRFC.title = "Size of the global MPR set (RFC)"
        mprRFC.outFile = "/tmp/mpr-rfc"
        mprRFC.xAxisLabel = "snapshot"
        link.legendPosition = "lower right"

        mprLq.x.append(v["MPRlq"]["MPR"]["x"])
        mprLq.y.append((v["MPRlq"]["MPR"]["y"], n))
        mprLq.title = "Size of the global MPR set (lq)"
        mprLq.outFile = "/tmp/mpr-lq"
        mprLq.xAxisLabel = "snapshot"
        link.legendPosition = "lower right"

        mprRobustness.x.append(v["MPRRFC"]["ROBUSTNESS"]["x"])
        mprRobustness.y.append((v["MPRRFC"]["ROBUSTNESS"]["y"], n+"-RFC"))
        mprRobustness.x.append(v["MPRlq"]["ROBUSTNESS"]["x"])
        mprRobustness.y.append((v["MPRlq"]["ROBUSTNESS"]["y"], n+"-lq"))
        mprRobustness.title = "Robustness metric of the MPR sub-graph"
        mprRobustness.outFile = "/tmp/robustness"
        mprRobustness.xAxisLabel = "Broken links"
        mprRobustness.yAxisLabel = "Robustness"
        mprRobustness.legendPosition = "aside"

        closeness.x.append(v["CENTRALITY"]["CLOS"]["x"])
        closeness.y.append((v["CENTRALITY"]["CLOS"]["y"], n))
        closeness.title = "Group closeness centrality"
        closeness.outFile = "/tmp/closeness"
        closeness.xAxisLabel = "Group Size"
        closeness.yAxisLabel = "Group Closeness"
        closeness.legendPosition = "upper right"

        closenessV.x.append(range(len(v["CENTRALITY"]["CLOSV"])))
        closenessV.y.append((v["CENTRALITY"]["CLOSV"], n))
        closenessV.title = "Group closeness centrality variation"
        closenessV.outFile = "/tmp/closeness-variation"
        closenessV.xAxisLabel = "snapshot"
        closenessV.yAxisLabel = "Group Closeness (5 nodes)"
        closenessV.legendPosition = "lower center"

        betweenness.x.append(v["CENTRALITY"]["BET"]["x"])
        betweenness.y.append((v["CENTRALITY"]["BET"]["y"], n))
        betweenness.title = "Group betweenness centrality"
        betweenness.outFile = "/tmp/betweenness"
        betweenness.xAxisLabel = "Group Size"
        betweenness.yAxisLabel = "Group Betweenness"
        betweenness.legendPosition = "aside"

        betweennessD.x.append(v["CENTRALITY"]["BETD"]["x"])
        betweennessD.y.append((v["CENTRALITY"]["BETD"]["y"], n))
        betweennessD.title = "Group betweenness (highest degree)"
        betweennessD.outFile = "/tmp/betweenness-estimation"
        betweennessD.xAxisLabel = "Group Size"
        betweennessD.yAxisLabel = "difference from the best group Betweenness (%)"
        betweennessD.legendPosition = "upper right"

    etx.plotData()
    link.plotData()
    mprLq.plotData()
    mprRFC.plotData()
    mprRobustness.plotData()
    closeness.plotData()
    betweenness.plotData()
    betweennessD.plotData()
    closenessV.plotData()
        



        
        




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
    if C.loadDb != "":
        engine = create_engine(C.loadDb)
        sessionFactory = sessionmaker(bind=engine, autocommit=True)
        localSession = scoped_session(sessionFactory)
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

    centFolder = C.resultDir+"/CENTRALITY"
    try:
        os.makedirs(centFolder)
    except OSError:
        try:
            os.mkdir(C.resultDir+"/backup/")
        except:
            pass
        now =  datetime.now()
        newFolder = C.resultDir+"backup/CENTRALITY"+str(now.month)+\
                str(now.day)+str(now.hour)+\
                str(now.minute)+str(now.second)
        shutil.move(C.resultDir+"/CENTRALITY", newFolder)
        os.mkdir(C.resultDir+"/CENTRALITY")


    parsers = []
    for net in data.rawData:
        q = Queue()
        parser = dataParser(net, q)
        p = Process(target=parser.run)
        parsers.append((net, p, q))
        p.start()

    for (n,p,q) in parsers:
        p.join()

    extractDataSeries(parsers)

    f = open(C.resultDir+"/logfile.txt", "w")
    runTime = datetime.now() - startTime - loadTime
    print >> f,  data.printSummary()
    print >> f, logString
    print >> f, "Whiskers = q3+/-1.5*IQR"
    print >> f, "loadTime =", loadTime, "runTime =", runTime
    f.close()


