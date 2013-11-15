#!/usr/bin/python

from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Float
from sqlalchemy import create_engine
from random import random
import sys

Base = declarative_base()

class scan(Base):
    __tablename__ = 'scan'
    Id = Column(Integer, primary_key=True)
    time = Column(DateTime, default=datetime.utcnow)
    scan_type = Column(String(10), default="ETX")

class node(Base):
    __tablename__ = 'node'
    Id = Column(Integer, primary_key=True)
    primary_address = Column(String(20))#, primary_key=True)
    hna = Column(String(20))
    links_outgoing = relationship("link", order_by="link.Id", 
            primaryjoin = "link.from_node_Id == node.Id")
    links_ingoing = relationship("link", order_by="link.Id", 
            primaryjoin = "link.to_node_Id == node.Id")

class link(Base):
    __tablename__ = 'link'
    Id = Column(Integer,primary_key=True)
    # two nodes may have more than one link 
    multi_link_number = Column(Integer)
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

def addTestScan(session):
    newscan = scan()
    session.add(newscan)
    N = 5
    for i in range(N):
        session.merge(node(primary_address="192.168.0."+str(i)))
    nodes = session.query(node)
    for i in range(N):
        l = session.merge(link(scan_Id_r = newscan, multi_link_number=0,
            from_node_r = nodes[i], to_node_r = nodes[(i+1)%N]))
        session.merge(etx(etx_value=1+random(), link_r=l))
    session.commit()




if __name__ == '__main__':

    #engine = create_engine('mysql://testdb@localhost/testdb')
    database = 'sqlite:///../testdb/ninux.db'
    engine = create_engine(database)
    Base.metadata.create_all(engine)
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    addTestScan(session)

       

