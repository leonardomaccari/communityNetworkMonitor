from plugin import plugin

from dbmanager import *
#FIXME move this to dbmanager
from sqlalchemy.exc import  SQLAlchemyError
import logging
import networkx as nx

class ninux(plugin):
    logger = None
    userName = ""
    userPasswd = ""
    dbURL = ""
    dbPort = None
    dbName = ""
    pluginName = ""
    enabled = True

    def initialize(self, parser, lc):
        self.parser = parser
        self.enabled, logLevel, self.pluginName = plugin.baseInitialize(self, 
                parser, lc, __file__)
        self.logger = logging.getLogger(self.pluginName)
        self.logger.setLevel(logLevel)
        self.localSession = lc
        self.userName = self.parser.get('ninux', 'user')
        self.userPasswd = self.parser.get('ninux', 'passwd')
        self.dbURL = self.parser.get('ninux', 'url')
        self.dbPort = self.parser.get('ninux', 'port')
        self.dbName = self.parser.get('ninux', 'db')
    
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
        nodeshot_interface as difc join nodeshot_device as ddev join nodeshot_node
        as dnode where snode.id = sdev.node_id and sdev.id = sifc.device_id and
        sifc.id = link.from_interface_id and difc.id = link.to_interface_id and
        difc.device_id = ddev.id and ddev.node_id = dnode.id"""
    
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
        for [sid, sname, did, dname, etxValue] in q:
            g.add_node(sid)
            g.add_node(did)
            g.add_edge(sid,did,weight=etxValue)
        try:
            self.localSession.commit()
        except  SQLAlchemyError as e:
            self.logger.error("could not write to local db: "+e.message)
        addGraphToDB(g,self.localSession, newScan)
    
