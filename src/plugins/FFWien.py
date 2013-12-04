
import requests
from requests.exceptions import RequestException
from simplejson.decoder import JSONDecodeError

import gzip
import os
from datetime import timedelta
import networkx as nx
from BeautifulSoup import BeautifulSoup
from mechanize import Browser, URLError
from threading import Thread
import logging
from collections import defaultdict

from dbmanager import *
from plugin import plugin

class URLException(Exception):
    pass

class JSONException(Exception):
    pass

class FFWien(plugin):

    def __init__(self):
        Thread.__init__(self)

    def initialize(self, parser, lc):
        #FIXME remove the logs from requests library!
        self.localSession = lc
        self.parser = parser
        # FIXME extend this debug to all the plugins
        self.debug = True
        self.enabled, logLevel, self.pluginName = plugin.baseInitialize(self, 
                parser, __file__, lc)
        self.logger = logging.getLogger(self.pluginName)
        self.logger.setLevel(logLevel)
        try:
            self.baseTopoURL = self.parser.get('FFWien', 'baseTopoURL')
            self.baseJSONURL = self.parser.get('FFWien', 'baseJSONURL')
            self.reloadJSON = self.convertTime(self.parser.get('FFWien', 
                'reloadJSON'))
        except Error as e:
            self.logger.error("Could not initalize FFWien plugin. \'"+e+"\'")
            if self.enabled == True:
                sys.exit()
        try:
            self.period = plugin.convertTime(self, self.parser.get('ninux', 
                'period'))
        except:
            self.period = 300
  
    def checkJSONDump(self):
        """ check if there is a recent dump of the JSON database """
        #FIXME check the return of this function 
        lastScan = self.localSession.query(scan).order_by(scan.time).\
                filter(and_((scan.scan_type=="JSON"),
                    (scan.network==self.pluginName))).all()
        if lastScan != []:
            if datetime.now() - timedelta(
                    seconds=self.reloadJSON) <  lastScan[-1].time:
                return lastScan[-1]
        return None

    def getJSONDump(self):
        """ get a new dump of the JSON and store it in the database """
        #FIXME this should be done in a separte thread, or the getNewTopology should get not only one topology

        wirelessInterfacesURL = self.baseJSONURL+\
                "/api/FFM-Wireless_Interface"
        wiredInterfacesURL = self.baseJSONURL+\
                "/api/FFM-Wired_Interface"

        newScan = scan(scan_type="JSON", network=self.pluginName)
        self.localSession.add(newScan)

        try: # wrappers for get() and json() will raise exceptions on errors
            wirelessInterfaces = self.getURLWrapper(wirelessInterfacesURL)
            wiredInterfaces = self.getURLWrapper(wiredInterfacesURL)
            wirelessInterfacesDict = self.decodeJSON(wirelessInterfaces,
                    wirelessInterfacesURL)
            wiredInterfacesDict = self.decodeJSON(wiredInterfaces,
                    wiredInterfacesURL)
            wirelessIfNum = len(wirelessInterfacesDict['entries'])
            for ifu in wirelessInterfacesDict['entries'] + \
                wiredInterfacesDict['entries']:
                if wirelessIfNum > 0:
                    ifType = "WLAN"
                else:
                    ifType = "LAN"
                wirelessIfNum -= 1
                ifaceURL = self.baseJSONURL+ifu
                iface = self.getURLWrapper(ifaceURL)
                deviceJSON = self.decodeJSON(iface, ifaceURL)

                deviceURL = self.baseJSONURL+deviceJSON\
                            ['attributes']['left']['url']
                rels = deviceJSON['rels']

                ip4Links = ""
                for r in rels:
                    if "ip4_network_links" in r:
                        ip4Links = r
                        break
                if ip4Links == "":
                    self.logger.debug("Found an interface with no IP",
                            "address associated")
                    continue
    
                ip4NetworkLinksURL = self.baseJSONURL + ip4Links
                ip4NetworkLinks = self.getURLWrapper(ip4NetworkLinksURL)
                ipAddresses = []
                ip4NetworkJSON = self.decodeJSON(ip4NetworkLinks, 
                        ip4NetworkLinksURL)
                for l in ip4NetworkJSON['entries']:
                    ip4Link = self.baseJSONURL + l
                    relIfaceIP = self.getURLWrapper(ip4Link)
                    ip4JSON = self.decodeJSON(relIfaceIP, ip4Link)
                    ip4URL = self.baseJSONURL + ip4JSON['attributes']\
                            ['right']['url']
                    ip4Data = self.getURLWrapper(ip4URL)
                    ipJSON = self.decodeJSON(ip4Data, ip4URL)
                    ipAddress = ipJSON['attributes']['net_address']
                    ipAddresses.append(ipAddress)

                device = self.getURLWrapper(deviceURL)
                deviceJSON = self.decodeJSON(device, deviceURL)
                nodePid = deviceJSON['attributes']['node']['pid']
                # FIXME
                # this is not the node name, it is the device name
                # we need one more get for the node name
                #nodeName = deviceJSON['attributes']['name']
                presentNode = self.localSession.query(node).filter(and_(
                    (node.Id==nodePid), (node.scan_Id==newScan.Id))).\
                            first()
                if presentNode:
                    newNode = presentNode
                else:
                    # FIXME I didn't find a way to avoid this query
                    # merge() should be taking care of avoiding 
                    # double inserts, but I can't make it work
                    newNode = node(Id=nodePid, name="",
                            scan_Id_r=newScan)
                self.localSession.merge(newNode)
                self.localSession.flush()
                networkEntry = self.localSession.query(network).filter_by(
                        name=self.pluginName)
                # this entry must exist, we set it up at initialize
                for a in ipAddresses:
                    netAd  = a.split('/')
                    if netAd[0] == "":
                        self.logger.error('There is a bogus entry in" +\
                            "the DB", %s from %s',a,ip4NetworkLinksURL)
                    else:
                        host = netAd[0]
                    if len(netAd) < 2:
                        net = 32
                    else:
                        net = netAd[1]
                    presentAddress = self.localSession.query(
                            IPv4Address).filter_by(IPv4=a).first()
                    if not presentAddress:
                        newAddress = IPv4Address(IPv4=host, 
                            netmask = net,
                            node_Id=newNode.Id,
                            iface_type=ifType,
                            network_Id_r=networkEntry[0]) 
                        self.localSession.merge(newAddress)
        except URLException:
            return False
        except JSONException:
            return False
        return True

    def buildIPMapping(self, lastScan):
        """ get mappings IP<->node from the last JSON scan in the DB """

        IPDevMatch = self.localSession.query(IPv4Address.IPv4,
                node.Id, IPv4Address.iface_type).filter(
                and_((IPv4Address.node_Id==node.Id), \
                        (node.scan_Id==lastScan.Id)))
        IPDevMap = {}
        nodeIPMap = defaultdict(dict)
        for IP, nodeId, ifType in IPDevMatch:
            IPDevMap[IP] = [nodeId, ifType]
            nodeIPMap[nodeId][IP] = ifType
        self.IPAddressToNode = IPDevMap
        self.nodeToIPAddress = nodeIPMap
        self.logger.info("Done reloading the DB")

    def getURLWrapper(self, url):
        """ Wrapper to catch exceptions when requesting a url """
        try:
            ret = requests.get(url)
        except RequestException as e:
            self.logger('Could not get URL %s, \'%s\'', url, e.message)
            raise URLException
        return ret

    def decodeJSON(self, data, url):
        """ Wrapper to catch exceptions when parsing JSON """
        try:
            ret = data.json()
        except JSONDecodeError as e:
            self.logger.error('Could not decode JSON from %s: %s',
                    url, e.message) 
            raise JSONException
        return ret

    def getLastEntry(self, url, browser):
        """ get one entry from an HTML table list if it is newer 
        than prevEntry. Format is from graz FF site """

        try:
            page = browser.open(url)
        except :
            if url == None:
                url = "(empty url)"
            self.logger.error("Could not read url "+url)
            return None, None
        html = page.read()
        soup = BeautifulSoup(html)
        link = soup.findAll('a')
        if len(link) == 0:
            logger.error('No links in the page: %s', url)
            return None, None
        rowDate = datetime.strptime(link[-1].string, "topo-%Y-%m-%d-%H:%M.tsv.gz")
        return url+link[-1].string, rowDate
        
    
    def getNewTopology(self):
        """ download the topology file from FFWien website """

        self.logger.info('Getting the latest topology')
        lastDateFromDB = self.localSession.query(scan).filter(and_(
                (scan.scan_type=="ETX"),(scan.network==self.pluginName))).\
                        order_by(desc(scan.time)).limit(1).first()

        if lastDateFromDB == None:
            lastDate = datetime.strptime("2000-Nov-01 12:00:00", 
                    "%Y-%b-%d %H:%M:%S")
        else:
            lastDate = lastDateFromDB.time
        mech = Browser()
        fileLink, newDate = self.getLastEntry(self.baseTopoURL, mech)

        if fileLink == None:
            return

        if newDate > lastDate:
            try:
                f = mech.retrieve(fileLink)
            except URLError:
                self.logger.error("Could not get link %s", filelink)
                return
        else:
            self.logger.info("Did not find a new topology file")
            return

        self.logger.info("Ok, parsing the topology")
        olsrDump = gzip.open(f[0])
        newScan = scan(network=self.pluginName) 
        self.localSession.add(newScan)
        G = nx.Graph()


        numUnknown ={}
        if self.debug == True:
            numWireless = 0
            numWired = 0
            numMixed = 0
            numEtxOne = 0
            numMixedNodes = 0
            multiLinks = defaultdict(int)
            reverseLinks = defaultdict(int)
            graph = defaultdict(lambda:defaultdict(lambda:defaultdict()))

        # the first lines are a headers
        for l in olsrDump.readlines()[2:]:
            try:
                ips, ipd, lq, rlq, etx = l.split("\t")
            except ValueError:
                if len(l) > 2:
                    self.logger.error("There was an error in the topology file line: %s", l)
                    return
                else:
                    continue
            if ips not in self.IPAddressToNode:
                self.logger.info("Skipped node IP %s not in the DB",ips)
                numUnknown[ips] = ""
                continue
            else:
                #FIXME name missing
                G.add_node(self.IPAddressToNode[ips][0], name="")

            if ipd not in self.IPAddressToNode:
                self.logger.info("Skipped node IP %s not in the DB",ipd)
                numUnknown[ipd] = ""
                continue
            else:
                #FIXME name missing
                G.add_node(self.IPAddressToNode[ipd][0], name="")
            src = self.IPAddressToNode[ips][0]
            dst = self.IPAddressToNode[ipd][0]
            srcType = self.IPAddressToNode[ips][1]
            dstType = self.IPAddressToNode[ipd][1]

            # NOTE multi-edges are squashed on thebest one

            #FIXME once debugging is over, do just one big if
            if etx != "INFINITE\n" and dstType == "LAN"\
                    and srcType == "LAN":
                        if self.debug == True:
                            numWired += 1
                            if dst in G[src]:
                                multiLinks["LL"] += 1
                            if src in G[dst]:
                                reverseLinks["LL"] += 1
                            graph[src][dst][etx] = "L"
    
                        if dst in G[src] and G[src][dst]['weight'] < etx:
                            continue
                        G.add_edge(src, dst, link_type=srcType,
                                weight=etx)

            elif etx != "INFINITE\n" and dstType == "WLAN" and\
                    srcType == "WLAN":
                        if self.debug == True:
                            if dst in G[src]:
                                multiLinks["WW"] += 1
                            if src in G[dst]:
                                reverseLinks["WW"] += 1
                            numWireless += 1
                            graph[src][dst][etx] = "W"
    
                        if dst in G[src] and G[src][dst]['weight'] < etx:
                            continue
                        G.add_edge(src, dst, link_type=dstType,
                                weight=etx)
            elif etx != "INFINITE\n" and dst != src:
                        if self.debug == True:
                            if dst in G[src]:
                                multiLinks["M"] += 1
                            if src in G[dst]:
                                reverseLinks["M"] += 1
                            numMixed += 1
                            if float(etx) == 1.0:
                                numEtxOne += 1
                            graph[src][dst][etx] = "M"

                        if dst in G[src] and G[src][dst]['weight'] < etx:
                            continue
                        G.add_edge(src, dst, link_type="MIXED",
                                weight=etx)
        if self.debug == True:
            allWired = 0
            allWireless = 0
            for node, ips in self.nodeToIPAddress.items():
                ifType = {}
                for ip,kind in ips.items():
                    ifType[kind] = ""
                if len(ifType) > 1:
                    numMixedNodes += 1
                elif len(ifType) == 1:
                    if "WLAN" in ifType:
                        allWireless += 1
                    elif "LAN" in ifType:
                        allWired += 1
                    else:
                        print "human error! human error!"

            print "Nodes", len(self.nodeToIPAddress), numMixedNodes,\
                    allWired, allWireless, len(numUnknown)
            print "Links", numWired, numWireless, numMixed, numEtxOne, multiLinks, reverseLinks

            numLinks = 0
            numDoubleLinks = 0
            numReverseLinks = 0
            for s in graph:
                for d in graph[s]:
                    numLinks += 1
                    if len(graph[s][d]) != 1:
                        numDoubleLinks += 1
                    if s in graph[d]:
                        numReverseLinks += 1
                    
            print "OLSR", numLinks, numDoubleLinks, numReverseLinks
        addGraphToDB(G, self.localSession, newScan)
        olsrDump.close()
        self.logger.info("Ok, completed the topology download")
        os.remove(f[0])
    
    def getStats(self):
        
        #logging.disable("requests")
        #reqL = logging.getLogger("requests")
        ##reqL.disable()

        lastScan = self.checkJSONDump() 
        if lastScan == None:
            self.logger.info('Re-building IP mapping from JSON')
            if self.getJSONDump():
                self.logger.info('Ok, rebuild the JSON structure')
                lastScan = self.checkJSONDump() 
            else:
                return
        else:
            self.logger.info('Getting IP mapping from database')
            self.buildIPMapping(lastScan)
            self.getNewTopology()
        
