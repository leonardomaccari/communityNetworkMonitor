from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Float
from sqlalchemy import create_engine, desc
from datetime import datetime
from random import random

Base = declarative_base()

class scan(Base):
    __tablename__ = 'scan'
    Id = Column(Integer, primary_key=True)
    time = Column(DateTime, default=datetime.utcnow)
    scan_type = Column(String(10), default="ETX")
    network = Column(String(10), default="NINUX")

class topo_file(Base):
    __tablename__= 'topo_file'
    file_url = Column(String(300), primary_key=True)
    time = Column(DateTime)
    scan_Id = Column(Integer, ForeignKey("scan.Id"))
    scan_Id_r = relationship(scan)

class node(Base):
    __tablename__ = 'node'
    Id = Column(Integer, primary_key=True)
    name = Column(String(50))
    links_outgoing = relationship("link", order_by="link.Id", 
            primaryjoin = "link.from_node_Id == node.Id")
    links_ingoing = relationship("link", order_by="link.Id", 
            primaryjoin = "link.to_node_Id == node.Id")

class link(Base):
    __tablename__ = 'link'
    Id = Column(Integer,primary_key=True)
    # two nodes may have more than one link 
    multi_link_number = Column(Integer, default=0)
    link_type = Column(String(10), default="WIRELESS") 

    from_node_Id = Column(Integer, ForeignKey("node.Id"))
    from_node_r = relationship(node, 
            primaryjoin="node.Id == link.from_node_Id")
            
    to_node_Id = Column(Integer, ForeignKey("node.Id"))
    to_node_r = relationship(node, 
            primaryjoin="node.Id == link.to_node_Id")

    scan_Id = Column(Integer, ForeignKey("scan.Id"))
    scan_Id_r = relationship(scan)

class etx(Base):
    __tablename__ = 'etx'
    link_Id = Column(Integer, ForeignKey('link.Id'), primary_key=True)
    etx_value = Column(Float)
    link_r = relationship(link)

def addTestScan(session, numNodes, randomize):
    newscan = scan()
    session.add(newscan)
    for i in range(numNodes):
        session.merge(node(name="test-node-"+str(i)))
    if randomize:
        ip = int((256-numNodes)*random())
        session.merge(node("test-node-" + str(numNodes+ip)))
    nodes = session.query(node)
    for i in range(numNodes):
        l = session.merge(link(scan_Id_r = newscan, multi_link_number=0,
            from_node_r = nodes[i], to_node_r = nodes[(i+1)%numNodes]))
        session.merge(etx(etx_value=1+random(), link_r=l))
    if randomize:
        r1 = int(random()*numNodes)
        r2 = int(random()*numNodes)
        l = session.merge(link(scan_Id_r = newscan, multi_link_number=0,
            from_node_r = nodes[r1], to_node_r = nodes[r2]))
        session.merge(etx(etx_value=1+random(), link_r=l))
    session.commit()


def addGraphToDB(graph, localSession, scanId):
    nodes = {}
    for edge in graph.edges(data=True):
        sid = edge[0]
        did = edge[1]
        etxValue = edge[2]['weight']
        # check if we have already scanned the source node
        if sid not in nodes.keys():
            # it's an unscanned new node, is it in the database?
            presentNode = localSession.query(node).filter_by(name=sid).first()
            if not presentNode:
                # not in the db, create new node
                tmps = node(name=sid)
                nodes[sid] = tmps 
            else:
                nodes[sid] = presentNode
                tmps = nodes[sid]
        else:
            # yes we scanned it
            tmps = nodes[sid]

        if did not in nodes.keys():
            presentNode = localSession.query(node).filter_by(name=did).first()
            if not presentNode:
                # not in the db, create new node
                tmpd = node(name=did)
                nodes[did] = tmpd 
            else:
                nodes[did] = presentNode
                tmpd = nodes[sid]
        else:
            # yes we scanned it
            tmpd = nodes[did]

        newLink = link(from_node_r=tmps, to_node_r=tmpd, scan_Id_r=scanId)
        newEtx = etx(link_r=newLink, etx_value=etxValue)
        localSession.add(newLink)
        localSession.add(newEtx)
    localSession.commit()
    #FIXME should return something here

def initializeDB(parser):

    database =  parser.get('main', 'localdb')
    engine = create_engine(database)
    Base.metadata.create_all(engine)
    DBSession = sessionmaker(bind=engine)
    localSession = DBSession()
    return localSession

