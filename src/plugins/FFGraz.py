from dbmanager import *
from mechanize import Browser
from plugin import plugin
from BeautifulSoup import BeautifulSoup
import gzip
import pygraphviz as pg
import networkx as nx
import collections
import logging
 
def tree():
    return collections.defaultdict(tree)

class FFGraz(plugin):
    localSession = None
    logger = None
    url = ""
    logger = None
    pluginName = ""

    def initialize(self, parser, lc):
        self.localSession = lc
        self.url=parser.get('FFGraz', 'baseTopoURL')
        self.logger = logging.getLogger(self.pluginName)
        logLevel, self.pluginName = plugin.baseInitialize(self, parser, 
                lc, __file__)
        self.logger.setLevel(logLevel)

    def getStats(self):
        """ Get the topology from FFGraz and elaborate it, returns a weighted 
        graph"""
    
        mech = Browser()
        mech.set_handle_robots(False)
        lastDateFromDB = self.localSession.query(topo_file).\
            order_by(desc(topo_file.time)).limit(1).all()
    
        if len(lastDateFromDB) == 0:
            lastDate = datetime.strptime("2010-Nov-01 12:09:12", "%Y-%b-%d %H:%M:%S")
        else:
            lastDate = lastDateFromDB[0].time
    
        newScan = scan(network="FFGRAZ")
    
        #FIXME get date of last scan of Graz
        year, newDate = self.getLastEntry(lastDate, self.url, mech)
        month, newDate = self.getLastEntry(lastDate, year, mech)
        fileName, newDate = self.getLastEntry(lastDate, month, mech)
        
        topoFile = mech.retrieve(fileName)
        f = gzip.open(topoFile[0])
        
        graphString = ""
        for l in f:
            graphString += l 
        G = pg.AGraph(encoding='UTF-8')
        Ga = G.from_string(graphString)
        pG = nx.from_agraph(Ga)
        toBeRemoved = []
        for e in pG.edges(data=True):
            if str(e[2]['label']) != 'HNA': 
                if e[2]['label'] == 'INFINITE':
                    e[2]['weight'] = 1000
                else:
                    e[2]['weight'] = float(e[2]['label'])
            else:
                toBeRemoved.append([e[0], e[1]])
        for l in toBeRemoved:
            pG.remove_edge(l[0], l[1])
        simpleG = self.aggregateNodesByName(pG)
        # remove self-links
        for e in simpleG.edges():
            if e[0] == e[1]:
                simpleG.remove_edge(e[0], e[1])
        newFile = topo_file(file_url=fileName, scan_Id_r=newScan, time=newDate)
        self.localSession.add(newFile)
        self.localSession.commit()
        addGraphToDB(simpleG, self.localSession, newScan)
    
    def aggregateNodesByName(self, graph):
        """ this function takes a graph with node names of the kind x.y.z and
        aggregates nodes. It will check if two nodes x1.y.z and x2.y.z have an ETX
        != 1, in this case they will be aggregated, x2.y.z will disappear and x1.y.z
        will get its links """
    
        p = tree() # a tree structure addressable as dict of dicts
        nodesAggregation = {}
        for e in graph.nodes():
            # initialize the aggregation function
            nodesAggregation[e] = e
            nameParts = e.split(".")
            if len(nameParts) == 3: # forget about nodes different from "x.y.z"
                # build the tree, root node is a dict
                p[nameParts[2]][nameParts[1]][nameParts[0]] = ""
    
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
                aggregatedG.add_node(ags)
            if agd not in aggregatedG:
                aggregatedG.add_node(agd)
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
    
    def getLastEntry(self, prevEntryDate, url, browser):
        """ get one entry from an HTML table list if it is newer 
        than prevEntry. Format is from graz FF site"""
        try:
            page = browser.open(url)
        except :
            return ""
        html = page.read()
        soup = BeautifulSoup(html)
        table = soup.find('table')
        #FIXME we need a logging system here, and to handle return parameters on
        # errors
        if len(table) == 0:
            print >> sys.stderr, "No table found in", url
            return "", ""
    
        rowLink = "" 
        row = table.findAll('tr')[-1]
        if len(row) == 0:
            print >> sys.stderr, "No row found in table in", url
            return "", ""
    
        for col in row:
            if col['class'] == 'm':
                rowDate = datetime.strptime(col.string, "%Y-%b-%d %H:%M:%S")
            if col['class'] == 'n':
                    link = col.findAll('a')
                    if len(link) != 0:
                        rowLink = url+link[0].get('href')
        if rowDate > prevEntryDate:
            return rowLink, rowDate
        return "", rowDate

