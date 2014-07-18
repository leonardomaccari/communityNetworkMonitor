
# @copyright Leonardo Maccari: leonardo.maccari@unitn.it
# released under GPLv3 license

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
import urllib2

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
        self.localSession = lc
        self.parser = parser
        self.debug = False # just a shortcut
        self.pseudonymDumpFile = None
        self.pseudonymFile = None
        self.ownerPseudonymDict = {}
        self.enabled, logLevel, self.pluginName = plugin.baseInitialize(self, 
                parser, __file__, lc)
        if logLevel < logging.INFO:
            self.debug = True
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
            self.period = self.convertTime(self.parser.get('FFWien', 'period'))
        except:
            self.period = 300

    def checkJSONDump(self):
        """ check if there is a recent dump of the JSON database """
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

        wirelessInterfacesURL = self.baseJSONURL+\
                "/api/FFM-Wireless_Interface"
        wiredInterfacesURL = self.baseJSONURL+\
                "/api/FFM-Wired_Interface"

        if self.myCrypto.disabled:
            newScan = scan(scan_type="JSON", network=self.pluginName)
        else:
            newScan = scan(scan_type="JSON", network=self.pluginName, encrypted=True)
        self.localSession.add(newScan)

        try: # wrappers for get() and json() will raise exceptions on errors
            wirelessInterfaces = self.getURLWrapper(wirelessInterfacesURL)
            wiredInterfaces = self.getURLWrapper(wiredInterfacesURL)
            wirelessInterfacesDict = self.decodeJSON(wirelessInterfaces,
                    wirelessInterfacesURL)
            wiredInterfacesDict = self.decodeJSON(wiredInterfaces,
                    wiredInterfacesURL)
            print wirelessInterfacesDict
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
                nodeURL = self.baseJSONURL + "/api/FFM-Node/" + str(nodePid)
                nodeData = self.getURLWrapper(nodeURL)
                nodeJSON = self.decodeJSON(nodeData, nodeURL)
                nodeName = nodeJSON['attributes']['name']
                nodeLat = 0
                nodeLon = 0
                nodeOwner = -1
                nodeManager = nodeJSON['attributes']['manager']['pid']

                if 'owner' in nodeJSON['attributes']:
                    nodeOwner = nodeJSON['attributes']['owner']['pid']

                if "position" in nodeJSON['attributes']:
                    nodeLat = nodeJSON['attributes']['position']['lat']
                    nodeLon = nodeJSON['attributes']['position']['lon']

                presentNode = self.localSession.query(node).filter(and_(
                    (node.Id==nodePid), (node.scan_Id==newScan.Id))).\
                            first()
                if presentNode:
                    newNode = presentNode
                else:
                    # TODO I didn't find a way to avoid this query
                    # merge() should be taking care of avoiding 
                    # double inserts, but I can't make it work
                    newNode = node(Id=nodePid, name=nodeName, owner=nodeOwner,
                            lat=nodeLat, lon=nodeLon, manager=nodeManager, 
                            scan_Id_r=newScan,)
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
            ret = requests.get(url, verify=False)
        except RequestException as e:
            self.logger.error('Could not get URL %s, \'%s\'', url, e.message)
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

    def getLastEntries(self, url, lastDate):
        """ get all entries from an HTML table list if it is newer 
        than prevEntry. Format is from graz FF site """

        mech = Browser()
        mech.set_handle_robots(False)
        try:
            page = mech.open(url)
        except urllib2.HTTPError:
            if url == None:
                url = "(empty url)"
            self.logger.error("Could not read url "+url)
            return []
        html = page.read()
        soup = BeautifulSoup(html)
        link = soup.findAll('a')
        if len(link) == 0:
            logger.error('No links in the page: %s', url)
            return []
        returnLinks = []

        for l in link:
            try:
                date = datetime.strptime(l.string, "topo-%Y-%m-%d-%H:%M.tsv.gz")
            except ValueError:
                continue
            if date > lastDate:
                returnLinks.append(url+l.string)
            else:
                break

        return returnLinks


    def getNewTopology(self):
        """ download the topology file from FFWien website """

        self.logger.info('Getting the latest topology file list')
        lastDateFromDB = self.localSession.query(scan).filter(and_(
                (scan.scan_type=="ETX"),(scan.network==self.pluginName))).\
                        order_by(desc(scan.time)).limit(1).first()

        if lastDateFromDB == None:
            lastDate = datetime.strptime("2000-Nov-01 12:00:00", 
                    "%Y-%b-%d %H:%M:%S")
        else:
            lastDate = lastDateFromDB.time
        linkList = self.getLastEntries(self.baseTopoURL, lastDate)

        if linkList == []:
            self.logger.info("No new topology files from last scan")
            return
        self.logger.info('Need to parse %d files', len(linkList))
        for fileLink in linkList:
            self.parseTopologyFile(fileLink)
            if lastDateFromDB == None:
                self.logger.info('breaking')
                # if this is the first ETX scan, we may dowload hundreds of
                # topologies. Avoid this, exit after the first one
                # FIXME as with the FFGraz plugin, this can be parametrized
                # and should be put in the base configuration parameters
                break

    def parseTopologyFile(self, fileLink):
        """ download and parse the topology file """

        mech = Browser()
        mech.set_handle_robots(False)
        try:
            f = mech.retrieve(fileLink)
        except URLError:
            self.logger.error("Could not get link %s", filelink)
            return
        self.logger.info("Ok, parsing the topology from url %s", fileLink)
        olsrDump = gzip.open(f[0])
        newDateString = fileLink.split("/")[-1]
        newDate = datetime.strptime(newDateString, "topo-%Y-%m-%d-%H:%M.tsv.gz")
        newScan = scan(network=self.pluginName, time=newDate) 
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
                self.logger.debug("Skipped node IP %s not in the DB",ips)
                numUnknown[ips] = ""
                continue
            else:
                G.add_node(self.IPAddressToNode[ips][0])

            if ipd not in self.IPAddressToNode:
                self.logger.debug("Skipped node IP %s not in the DB",ipd)
                numUnknown[ipd] = ""
                continue
            else:
                G.add_node(self.IPAddressToNode[ipd][0])
            src = self.IPAddressToNode[ips][0]
            dst = self.IPAddressToNode[ipd][0]
            srcType = self.IPAddressToNode[ips][1]
            dstType = self.IPAddressToNode[ipd][1]

            # NOTE multi-edges are squashed on thebest one

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

                    self.logger.debug("Nodes %d,%d,%d,%d,%d",
                            len(self.nodeToIPAddress), numMixedNodes, allWired,
                            allWireless, len(numUnknown))
                    self.logger.debug("Links %d, %d, %d, %d, %d, %d, %d, %d",
                            numWired, numWireless, numMixed, numEtxOne,
                            multiLinks, reverseLinks)

            numLinks = 0
            numDoubleLinks = 0
            numReverseLinks = 0
            for s in graph:
                for d in graph[s]:
                    numLinks += 1
                    if len(graph[s][d]) != 1:
                        numDoubleLinks += 1
                    if d in graph and s in graph[d]:
                        numReverseLinks += 1

            self.logger.debug("OLSR %d, %d, %d", numLinks, 
                    numDoubleLinks, numReverseLinks)
        #FIXME Need to add owners and emails here 
        addGraphToDB(G, self.localSession, newScan, self.myCrypto)
        olsrDump.close()
        self.logger.info("Ok, completed the topology download:"+\
                "%d nodes, %d links, %d unknown", len(G.nodes()), 
                len(G.edges()),len(numUnknown))
        os.remove(f[0])

    def getStats(self):
        """ main function called by run() in plugin class """

        for name,lg in logging.Logger.manager.loggerDict.items():
            if "urllib" in name:
                lg.setLevel(logging.ERROR)
        
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

