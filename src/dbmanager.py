
# @copyright Leonardo Maccari: leonardo.maccari@unitn.it
# released under GPLv3 license

from sqlalchemy.orm import relationship, sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy import String, DateTime, Float, Boolean
from sqlalchemy import create_engine, and_, desc
from sqlalchemy.exc import OperationalError
from datetime import datetime
from random import random
import sys

Base = declarative_base()

# objects representing the local database 
class scan(Base):
    __tablename__ = 'scan'
    Id = Column(Integer, primary_key=True)
    time = Column(DateTime, default=datetime.now)
    scan_type = Column(String(10), default="ETX")
    network = Column(String(10), default="NINUX")

class person(Base):
    __tablename__= 'person'
    Id = Column(Integer, primary_key=True)
    network = Column(String(10), default="NINUX")
    username = Column(String(256), default="")
    email = Column(String(256), default="")

class topo_file(Base):
    __tablename__= 'topo_file'
    scan_Id = Column(Integer, ForeignKey("scan.Id"), primary_key=True)
    time = Column(DateTime)
    file_url = Column(String(300))
    scan_Id_r = relationship(scan)

class node(Base):
    __tablename__ = 'node'
    Id = Column(String(50), primary_key=True)
    scan_Id = Column(Integer, ForeignKey('scan.Id'), primary_key=True)
    name = Column(String(50))
    owner_Id = Column(Integer, ForeignKey(person.Id))
    manager_Id = Column(Integer, ForeignKey(person.Id))
    lat = Column(Float)
    lon = Column(Float)
    scan_Id_r = relationship(scan)
    owner_Id_r = relationship(person, 
            primaryjoin="person.Id == node.owner_Id")
    manager_Id_r = relationship(person,
            primaryjoin="person.Id == node.manager_Id")

class network(Base):
    __tablename__ = 'network'
    Id = Column(Integer, primary_key=True)
    name = Column(String(50))
    description = Column(String(100))

class IPv4Address(Base):
    __tablename__ = 'ip_address'
    #TODO HNA and netmask is unused so far
    IPv4 = Column(String(15), primary_key=True)
    netmask = Column(String(2), primary_key=True)
    network_Id = Column(Integer, ForeignKey('network.Id'), primary_key=True)
    iface_type = Column(String(4), default="WLAN")
    node_Id = Column(Integer, ForeignKey('node.Id'))
    HNA = Column(Boolean())
    gateway = Column(Boolean())
    network_Id_r = relationship(network)
    # FIXME cascade == none, see fixme on FFWIEN
    node_Id_r = relationship(node, cascade='none')

class link(Base):
    __tablename__ = 'link'
    Id = Column(Integer,primary_key=True)
    # two nodes may have more than one link 
    multi_link_number = Column(Integer, default=0)
    link_type = Column(String(10), default="WLAN") 

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

def addGraphToDB(graph, localSession, scanId):
    """ transforms a nx graph in db entries """

    nodes = {}
    people = {}
    for edge in graph.edges(data=True):
        sid = edge[0]
        did = edge[1]
        sperson = (graph.node[edge[0]]["owner"],graph.node[edge[0]]["email"])
        dperson = (graph.node[edge[1]]["owner"],graph.node[edge[1]]["email"])
        etxValue = edge[2]['weight']
        if sperson not in people:
            # it's an unscanned new person, is it in the database?
            newSPerson = localSession.query(person).filter(
                    and_(person.username==sperson[0],
                        person.email==sperson[1])).first()
            if not newSPerson:
                # not in the db, create new person
                newSPerson = person(username=sperson[0], email=sperson[1])
            people[sperson] = newSPerson
        else:
            newSPerson = people[sperson]

        if dperson not in people:
            # it's an unscanned new person, is it in the database?
            newDPerson = localSession.query(person).filter(\
                    and_(person.username==dperson[0],
                        person.email==dperson[1])).first()
            if not newDPerson:
                # not in the db, create new person
                newDPerson = person(username=dperson[0], email=dperson[1])
            people[dperson] = newDPerson
        else:
            newDPerson = people[dperson]

        #TODO  I have get its Id 
        if sid not in nodes.keys():
            # it's an unscanned new node, is it in the database?
            presentNode = localSession.query(node).filter(\
                    and_(node.Id==sid, node.scan_Id==scanId.Id)).first()
            if not presentNode:
                # not in the db, create new node
                sname = ""
                if 'name' in graph.node[sid]:
                    sname = graph.node[sid]['name']
                tmps = node(Id=sid, scan_Id_r=scanId, name=sname,
                        owner_Id_r=newSPerson)
                nodes[sid] = tmps 
            else:
                nodes[sid] = presentNode
                tmps = nodes[sid]
        else:
            # yes we scanned it
            tmps = nodes[sid]

        if did not in nodes.keys():
            presentNode = localSession.query(node).filter(\
                    and_(node.Id==did, node.scan_Id==scanId.Id)).first()
            if not presentNode:
                # not in the db, create new node
                dname = ""
                if 'name' in graph.node[did]:
                    dname = graph.node[did]['name']
                tmpd = node(Id=did, scan_Id_r=scanId, name=dname, 
                        owner_Id_r=newDPerson)
                nodes[did] = tmpd 
            else:
                nodes[did] = presentNode
                tmpd = nodes[sid]
        else:
            # yes we scanned it
            tmpd = nodes[did]
        lt = ""
        if "link_type" in edge[2]:
            lt=edge[2]['link_type']
        newLink = link(from_node_r=tmps, to_node_r=tmpd, 
                scan_Id_r=scanId, link_type=lt)
        newEtx = etx(link_r=newLink, etx_value=etxValue)
        localSession.add(newEtx)
    #FIXME should return something here

def initializeDB(parser):
    #FIXME should catch errors here, do not put a default 
    # in the sqlite database? or add testdb to the git repo
    try:
        database =  parser.get('main', 'localdb')
        engine = create_engine(database)
        Base.metadata.create_all(engine)
    except OperationalError:
        print "ERROR: Could not open the local database. This may be to wrong"
        print "mySQL credentials or non-existent sqlite folder."
        print "Please check your configuration"
        sys.exit(1)
    sessionFactory = sessionmaker(bind=engine, autocommit=True)
    localSession = scoped_session(sessionFactory)
    return localSession

