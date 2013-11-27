#!/usr/bin/python

from ConfigParser import SafeConfigParser
from glob import glob
import sys
import logging
import time

import dbmanager
from plugins.plugin import plugin
from plugins.ninux import ninux 
from plugins.FFGraz import FFGraz
# import here future plugin code

def getConfig():
    """ parse config files in ../conf/ """
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

    threadList = []

    ffg = FFGraz()
    ffg.initialize(parser, localSession)
    nnx = ninux()
    nnx.initialize(parser, localSession)
    
    threadList.append(nnx)
    threadList.append(ffg)

    try:
        for i in threadList:
            i.daemon = True
            i.start()
        while True:
            for i in threadList:
                i.join(1)
        #ffg.start()
    except KeyboardInterrupt:
        for i in threadList:
            i.exitAll = True
        #raise
    waitTime = 20
    while True:
        a = sum([i.is_alive() for i in threadList])
        if a != 0:
            logger.info("Waiting %s seconds for %d threads to gracefully exit", waitTime, a)
        else:
            break
        if waitTime <= 0:
            logger.info("Ok, killing the threads")
            sys.exit()
        time.sleep(5)
        waitTime -= 5
    logger.info("Ok, all thread exited gracefully")


