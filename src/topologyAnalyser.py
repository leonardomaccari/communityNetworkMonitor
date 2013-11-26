#!/usr/bin/python

from ConfigParser import SafeConfigParser
from glob import glob
import sys
import logging

import dbmanager
from plugins.ninux import ninux 
from plugins.FFGraz import FFGraz
# import here future plugin code

def getConfig():
    parser = SafeConfigParser()
    pluginConfigFiles = glob("../conf/accesskeys/*.conf")
    mainConfigFile = glob("../conf/*.conf")
    return parser.read(pluginConfigFiles), parser.read(mainConfigFile),\
            parser

if __name__ == '__main__':

    pluginConfigFiles, mainConfigFile, parser = getConfig()
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

    if mainConfigFile == []:
        logger.critical("Could not read main config file")
        sys.exit(1)
    if pluginConfigFiles == []:
        logger.critical("Could not read config files")
        sys.exit(1)
    logger.info("Starting topologyAnalyser daemon")
    localSession = dbmanager.initializeDB(parser)
    ffg = FFGraz()
    ffg.initialize(parser, localSession)
    ffg.getStats()
    nnx = ninux()
    nnx.initialize(parser, localSession)
    nnx.getStats()
    #TODO make a thread to run multiple scan every X minutes

