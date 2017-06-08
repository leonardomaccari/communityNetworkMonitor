
# @copyright Leonardo Maccari: leonardo.maccari@unitn.it
# released under GPLv3 license

from mechanize import Browser
from threading import Thread
from ConfigParser import Error
import time
from BeautifulSoup import BeautifulSoup
from sqlalchemy import desc
import gzip
import pygraphviz as pg
import networkx as nx
import collections
import logging
import os
 
from plugin import plugin
from dbmanager import *

def tree():
    return collections.defaultdict(tree)

class FFGraz(plugin):

    def __init__(self):
        Thread.__init__(self)

    def initialize(self, parser, lc):
        self.localSession = lc
        self.parser = parser
        self.url=parser.get('FFGraz', 'baseTopoURL')
        self.pseudonymDumpFile = None
        self.pseudonymFile = None
        self.ownerPseudonymDict = {}
        self.enabled, logLevel, self.pluginName = plugin.baseInitialize(self, 
                parser, __file__, lc)
        self.logger = logging.getLogger(self.pluginName)
        self.logger.setLevel(logLevel)
        try:
            self.period = self.convertTime(self.parser.get('ninux', 'period'))
        except:
            import code
            code.interact(local=locals())
            self.period = 600

    def getStats(self, refillDB=True):
        """ Get the topology from FFGraz, elaborate it and store in the DB,
        refillDB=True will get one snapshot per month. 
        WARNING, this will download hundreds of files, it is usable only if
        you have a local mirror of the topologies
        """
        
        if self.enabled == False:
           self.logger.info(plugin.disabledMessage) 
           return
        self.logger.info("Getting data from FFGraz network")
        mech = Browser()
        mech.set_handle_robots(False)
        lastDateFromDB = self.localSession.query(topo_file).\
            order_by(desc(topo_file.time)).limit(1).all()
    
        if len(lastDateFromDB) == 0:
            lastDate = datetime.strptime("2000-Nov-01 12:00:00", 
                    "%Y-%b-%d %H:%M:%S")
        else:
            lastDate = lastDateFromDB[0].time
    
    
        if refillDB == False:
            fileName = None
            newDate = None

            # FIXME these variable names stink
            year, newDate = self.getLastEntry(self.url, mech)
            month, newDate = self.getLastEntry(year, mech)
            fileName, newDate = self.getLastEntry(month, mech)

            if fileName == None and newDate == None:
                return 

            if newDate <= lastDate:
                self.logger.error("Did not find an entry newer than "+\
                        str(lastDate))
                return
            self.addScan(mech, fileName, newDate)
        else:
            yearLinks  = self.getAllEntries(self.url, mech)
            for (l,y) in yearLinks:
                monthLinks = self.getAllEntries(l, mech)
                for (ml, m) in monthLinks:
                    link, date = self.getLastEntry(ml, mech)
                    self.logger.info("Added scan for file %s" % link)
                    self.addScan(mech, link, date)
         
 
    def addScan(self, mech, fileName, newDate):
        if self.myCrypto.disabled:
            newScan = scan(network=self.pluginName, time=newDate)
        else:
            newScan = scan(network=self.pluginName, time=newDate,
                    encrypted=True)
        try:
            topoFile = mech.retrieve(fileName)
        except: 
            self.logger.error("Could not retrieve file "+fileName)
            return
        try: 
            f = gzip.open(topoFile[0])
        except:
            self.logger.error("Could not open temporary file "+topoFile[0])
            return
        
        graphString = ""
        for l in f:
            graphString += l 
        G = pg.AGraph(encoding='UTF-8', directed=False)
        Ga = G.from_string(graphString)
        pG = nx.from_agraph(Ga)
        # note that FFGraz nodes are taken from the .dot file, 
        # so we don't have numerical IDs. In the DB the node_Id 
        # will be just the node name
        toBeRemoved = []
        for e in pG.edges(data=True):
            if str(e[2]['label']) != 'HNA': 
                if e[2]['label'] == 'INFINITE':
                    e[2]['weight'] = 1000
                else:
                    e[2]['weight'] = float(e[2]['label'])
            else:
                toBeRemoved.append([e[0], e[1]])
        for n in pG.nodes():
            for neigh in pG[n]:
                if n not in pG[neigh]:
                    toBeRemoved.append([n, neigh])

        for l in toBeRemoved:
            if (l[0], l[1]) in pG.edges():
                print "QQQ removed edge", l
                pG.remove_edge(l[0], l[1])
        simpleG = self.aggregateNodesByName(pG)
        for e in simpleG.edges():
            if e[0] == e[1]:
                simpleG.remove_edge(e[0], e[1])
        newFile = topo_file(file_url=fileName, scan_Id_r=newScan, time=newDate)
        self.localSession.add(newFile)
        #FIXME NEed to add owners and emails here
        addGraphToDB(simpleG, self.localSession, newScan, self.myCrypto)
        f.close()
        os.remove(topoFile[0])
        #FIXME split this function in pieces and add some proper logging for
        # begin/end of the function. Save the log message as a class plugin
        # variable so that all plugins use the same message
    
    def aggregateNodesByName(self, graph):
        """ this function takes a graph with node names of the kind x.y.z and
        aggregates nodes. It will check if two nodes x1.y.z and x2.y.z have an 
        ETX != 1, in this case they will be aggregated, x2.y.z will disappear 
        and x1.y.z will get its links """
    
        p = tree() # a tree structure addressable as dict of dicts
        nodesAggregation = {}
        for e in graph.nodes():
            # initialize the aggregation function
            nodesAggregation[e] = e
            nameParts = e.split(".")
            if len(nameParts) == 3: # forget about nodes different from "x.y.z"
                # build the tree, root node is a dict
                p[nameParts[2]][nameParts[1]][nameParts[0]] = ""
            else:
                print "WWW", e
    
        for node in p:
            nodeLeafs = []
            # recover each subtree
            for device in p[node]:
                for iface in p[node][device]:
                    nodeLeafs.append(iface+"."+device+"."+node)
            tmpLeafs = nodeLeafs[:]
            for l in nodeLeafs:
                # this node has already been aggregated in a previous run
                if nodesAggregation[l] != l:
                    continue
                #print l,":",
                for g in tmpLeafs[:]:
                    if l == g or nodesAggregation[g] != g:
                        continue
                    # l and g are in the same subtree, if they are neighbors 
                    # and weight == 1, we can suppress g
                    if g in graph[l] and graph[l][g][0]['weight'] == 1.0 \
                        and graph[g][l][0]['weight'] == 1.0:
                        #print g,
                        nodesAggregation[g] = l
                #print
        aggregatedG = nx.Graph()
        for s,d,data in graph.edges(data=True):
            ags = nodesAggregation[s]
            agd = nodesAggregation[d]
            if ags not in aggregatedG:
                aggregatedG.add_node(ags, name=str(ags))
            if agd not in aggregatedG:
                aggregatedG.add_node(agd, name=str(agd))
            print "XX", ags, agd, data
            aggregatedG.add_edge(ags,agd, weight=data['weight'])
    
        return aggregatedG
       
    def aggregateNodesByNameOld(self, graph):
        """ legacy function, just aggregates node based on the name """
    
        simpleG = nx.Graph()
        for e in graph.edges(data=True):
            nameParts = e[0].split(".")
            simpleG.add_node(nameParts[-1])
            nameParts = e[1].split(".")
            simpleG.add_node(nameParts[-1])
        
        for e in graph.edges(data=True):
            fromNode = e[0].split(".")[-1]
            toNode = e[1].split(".")[-1]
            if not simpleG.has_edge(fromNode, toNode) or \
                    simpleG[fromNode][toNode]['weight'] > e[2]['weight']:
                simpleG.add_edge(fromNode, toNode, weight=e[2]['weight'])
    
    def getLastEntry(self, url, browser):
        """ get the last entry from an HTML list """
        try:
            page = browser.open(url)
        except :
            if url == None:
                url = "(empty url)"
            self.logger.error("Could not read url "+url)
            return None, None
        html = page.read()
        soup = BeautifulSoup(html)
        linkList = soup.findAll('a')
        if len(linkList) == 0:
            self.logger.error("No link list found in "+url)
            return None, None
    
        
        if len(linkList) != 0:
            link = url+linkList[-1].get('href')
            rowText = linkList[-1].string.split(".")[0]
            try:
                rowDate = datetime.strptime(rowText, 
                                            "%Y-%m-%d_%H")
            except ValueError:
                try:
                    rowDate = datetime.strptime(rowText, 
                                            "%Y-%m-%d_%H_%M")
                except ValueError:
                    rowDate = rowText

        print link, rowDate
        return link, rowDate 

    def getAllEntries(self, url, browser):
        """ get one entry from an HTML table list if it is newer 
        than prevEntry. Format is from graz FF site"""
        try:
            page = browser.open(url)
        except :
            if url == None:
                url = "(empty url)"
            self.logger.error("Could not read url "+url)
            return None, None
        html = page.read()
        soup = BeautifulSoup(html)
        linkList = soup.findAll('a')
        if len(linkList) == 0:
            self.logger.error("No list of links found in "+url)
            return None, None
        links = [] 
        # skip 1 row of header
        for row in linkList[1:]:
            link = url+row.get('href')
            rowText = row.string
            try:
                rowDate = datetime.strptime(rowText, 
                                            "%Y-%b-%d_%H_%M.dot.gz")
            except ValueError:
                rowDate = rowText


            links.append([link, rowDate])
        return links
