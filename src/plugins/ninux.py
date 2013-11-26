from plugin import plugin

from dbmanager import *
import logging
import os

class ninux(plugin):
    logger = None
    userName = ""
    userPasswd = ""
    dbURL = ""
    dbPort = None
    dbName = ""
    pluginName = ""

    def initialize(self, parser, lc):
        self.parser = parser
        logLevel, self.pluginName = plugin.baseInitialize(self, parser, 
                lc, __file__)
        self.logger = logging.getLogger(self.pluginName)
        self.logger.setLevel(logLevel)
        self.localSession = lc
        self.userName = self.parser.get('ninux', 'user')
        self.userPasswd = self.parser.get('ninux', 'passwd')
        self.dbURL = self.parser.get('ninux', 'url')
        self.dbPort = self.parser.get('ninux', 'port')
        self.dbName = self.parser.get('ninux', 'db')
    
    def getNinuxStats(self):
    
        ninuxURL = 'mysql://'+self.userName+":"+self.userPasswd+"@"+\
        self.dbURL+":"+self.dbPort+"/"+self.dbName
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
    
        q = session.query("sid", "sname", "did", "dname", "etx_v").from_statement(
                etxQuery)
        try:
            c = len(q.all())
            if c==0:
                self.logger.error("no results from ninux DB!")
        except:
            self.logger.error("could not connect to ninux DB!")
        return 
        nodes = {}
        newScan = scan()
        self.localSession.add(newScan)
        for [sid, sname, did, dname, etxValue] in q:
            if sid not in nodes.keys():
                tmps = node(Id=int(sid))#, name=str(sname))
                nodes[sid] = tmps 
            else:
                tmps = nodes[sid]
            if did not in nodes.keys():
                tmpd = node(Id=int(did))#, name=str(dname))
                nodes[did] = tmpd 
            else:
                tmpd = nodes[did]
    
            newLink = link(from_node_r=tmps, to_node_r=tmpd, scan_Id_r=newScan)
            newEtx = etx(link_r=newLink, etx_value=etxValue)
            self.localSession.add(newLink)
            self.localSession.add(newEtx)
        try:
            self.localSession.commit()
        except:
            logger.error("could not write to local db")
    
