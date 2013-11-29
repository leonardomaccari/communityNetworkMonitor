from plugin import plugin
from threading import Thread
from ConfigParser import Error
from sqlalchemy.exc import  SQLAlchemyError
import logging
import networkx as nx
import time
import chardet

from dbmanager import *

class ninux(plugin):
    def __init__(self):
        Thread.__init__(self)
    def initialize(self, parser, lc):
        self.localSession = lc
        self.parser = parser
        self.enabled, logLevel, self.pluginName = plugin.baseInitialize(self, 
                parser, __file__, lc)
        self.logger = logging.getLogger(self.pluginName)
        self.logger.setLevel(logLevel)
        try:
            self.userName = self.parser.get('ninux', 'user')
            self.userPasswd = self.parser.get('ninux', 'passwd')
            self.dbURL = self.parser.get('ninux', 'url')
            self.dbPort = self.parser.get('ninux', 'port')
            self.dbName = self.parser.get('ninux', 'db')
        except Error as e:
            self.logger.error("Could not initalize ninux plugin. \'"+e+"\'")
            if self.enabled == True:
                sys.exit()
        try:
            self.period = plugin.convertTime(self, self.parser.get('ninux', 'period'))
        except:
            self.period = 300
    
    def getStats(self):
    
        if self.enabled == False:
            self.logger.info(plugin.disabledMessage)
            return
        ninuxURL = 'mysql://'+self.userName+":"+self.userPasswd+"@"+\
        self.dbURL+":"+self.dbPort+"/"+self.dbName

        self.logger.info("Getting data from Ninux network")
        engine = create_engine(ninuxURL)
        DBSession = sessionmaker(bind=engine)
        session = DBSession()
        ### FIXME this should be ORM-ed, not just raw SQL as it is now
        etxQuery = """select snode.id as sid, snode.name as sname, 
        dnode.id as did, dnode.name as dname, link.etx as etx_v
        from nodeshot_node as snode join nodeshot_device as sdev join
        nodeshot_interface as sifc join nodeshot_link as link join
        nodeshot_interface as difc join nodeshot_device as ddev join
        nodeshot_node as dnode where snode.id = sdev.node_id and sdev.id
        = sifc.device_id and sifc.id = link.from_interface_id and
        difc.id = link.to_interface_id and difc.device_id = ddev.id and
        ddev.node_id = dnode.id"""
    
        q = session.query("sid", "sname", "did", "dname", "etx_v").\
            from_statement(etxQuery)

        try:
            c = len(q.all())
            if c==0:
                self.logger.error("no results from ninux DB!")
                return 
        except:
            #FIXME add error message from the DB
            self.logger.error("could not connect to ninux DB!")
            return 

        newScan = scan(network="NINUX")
        self.localSession.add(newScan)
        g = nx.Graph()
        for [sid, sn, did, dn, etxValue] in q:
            # some strings in ninux DB are encoded not in weird
            # encodings, guess the encoding and convert
            sname = sn.decode(chardet.detect(sn)['encoding'])
            dname = dn.decode(chardet.detect(dn)['encoding'])
            g.add_node(sid, name=sname)
            g.add_node(did, name=dname)
            g.add_edge(sid,did,weight=etxValue)
        addGraphToDB(g, self.localSession, newScan)

