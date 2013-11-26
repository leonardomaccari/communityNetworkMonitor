#!/usr/bin/python

from ConfigParser import SafeConfigParser
from glob import glob
import sys
import logging

import dbmanager
from plugins.ninux import ninux 
from plugins.FFGraz import FFGraz

def getConfig():
    parser = SafeConfigParser()
    configFiles = glob("../conf/accesskeys/*.conf")
    configFiles += glob("../conf/*.conf")
    readFiles = parser.read(configFiles)
    if readFiles == []:
        print >> sys.stderr, "ERROR: No config file found in ",\
                "../conf/accesskeys/ !"
    return readFiles, parser

if __name__ == '__main__':

    configFiles, parser = getConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    sHandler = logging.StreamHandler()
    sHandler.setLevel(logging.INFO)
    fHandler = logging.FileHandler(parser.get('main', 'logfile'))
    fHandler.setLevel(logging.INFO)
    sFormatter = logging.Formatter('%(name)s: %(levelname)s, %(message)s')
    fFormatter = logging.Formatter('%(asctime)s, %(name)s: %(levelname)s, %(message)s')
    sHandler.setFormatter(sFormatter)
    fHandler.setFormatter(fFormatter)
    logger.addHandler(sHandler)
    logger.addHandler(fHandler)
    lg = logging.getLogger("puppa")
    lg.error('XX')
    logger.error('XXX')
    localSession = dbmanager.initializeDB(parser)
    #getNinuxStats(parser, localSession)
    #ffg = FFGraz()
    #ffg.initialize(parser, localSession)
    #ffg.getFFGrazStats()
    nn = FFGraz()
    nn.initialize(parser, localSession)
    nn.getStats()

    sys.exit()

    #TODO make a thread to run multiple scan every X minutes
    # separate the code in distinct files for each network, corresponding 
    # to each thread

