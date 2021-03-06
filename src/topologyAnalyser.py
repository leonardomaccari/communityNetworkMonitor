#!/usr/bin/python

#FIXME add copyright notice to every file

# @copyright Leonardo Maccari: leonardo.maccari@unitn.it
# released under GPLv3 license

from ConfigParser import SafeConfigParser
from glob import glob
import sys
import logging
import time
import getopt
import os
import signal
import requests
import datetime

import dbmanager
from plugins.ninux import ninux 
from plugins.FFGraz import FFGraz
from plugins.FFWien import FFWien
# import here future plugin code

def getConfig():
    """ parse config files in ../conf/ """
    parser = SafeConfigParser()
    pluginConfigFiles = glob("../conf/accesskeys/*.conf")
    mainConfigFile = glob("../conf/*.conf")
    return parser.read(pluginConfigFiles), parser.read(mainConfigFile),\
            parser



def parseArgs():
    """ argument parser """
    C = configuration()
    try:
        opts, args = getopt.getopt(sys.argv[1:], "dv:h")
    except getopt.GetoptError, err:
        # print help information and exit:
        print >> sys.stderr,  str(err)
        C.printUsage()
        sys.exit(2)
    for option,v in opts:
        if option in C.neededParamsNames.keys():
            optionValue = C.neededParamsNames[option]
            if optionValue[1] == True:
                C.neededParams[optionValue[0]] = optionValue[4](v)
            else:
                C.neededParams[optionValue[0]] = True
        elif option in C.optionalParamsNames.keys():
            optionValue = C.optionalParamsNames[option]
            if optionValue[1] == True:
                C.optionalParams[optionValue[0]] = optionValue[4](v)
            else:
                C.optionalParams[optionValue[0]] = True
        else:
            assert False, "unhandled option"

    if C.checkCorrectnes() == False:
        C.printUsage()
        sys.exit(1)
    return C


class configuration():
    """ configuration parameters storage class."""

    # the syntax is:
    #  "command line option"->[optionName, wantsValue, 
    #           defaultValue, usageMessage, type]
    # to add a parameter add a line in the needed/optional row, use
    # getParam("paramName") to get the result or check if it set.
    # optional parameters should always use False as default value, so they 
    # return False on getParam()

    neededParamsNames = {
            # no needed parameters
    }
    optionalParamsNames = {
            "-d":["daemonMode", False, False, "go to background, do not write to stdout", str],
            "-v":["verbosity", True, 1, "verbosity level 0-2", int],
            "-h":["help", False, False, "show the help", int],
            }
    defaultValue = False
    neededParams = {}
    optionalParams = {}

    def __init__(self):
        for pname, pvalue in self.neededParamsNames.items():
            self.neededParams[pvalue[0]] = pvalue[2]
        for pname, pvalue in self.optionalParamsNames.items():
            self.optionalParams[pvalue[0]] = pvalue[2]

    def checkCorrectnes(self):
        """ do some consistence checks here for the configuration parameters """
        if self.getParam("help") == True:
            return False
        return True

    def printUsage(self):
        """ print the usage of the program """
        print >> sys.stderr
        print >> sys.stderr, "usage:",
        print >> sys.stderr, "./topologyAnalyser.py:"
        for pname, pvalue in self.neededParamsNames.items():
            print >> sys.stderr, " ", pname, pvalue[3]
        for pname, pvalue in self.optionalParamsNames.items():
            print >> sys.stderr, " [",pname, pvalue[3], "]"

    def getParam(self, paramName):
        """ return a configuration parameter """
        for pname, pvalue in self.neededParamsNames.items():
            if pvalue[0] == paramName:
                return self.neededParams[paramName]
        for pname, pvalue in self.optionalParamsNames.items():
            if pvalue[0] == paramName:
                return self.optionalParams[paramName]
        print >> sys.stderr, "coding error: the",\
            paramName, "parameter does not exist"
        sys.exit(1)

    def printConf(self):
        """ just print all the configuration for debug """
        print ""
        for pname, pvalue in self.neededParams.items():
            print pname, pvalue
        for pname, pvalue in self.optionalParams.items():
            print pname, pvalue


def termHandler(signum, frame):
    """ try to leave time to the processess to exit, so they don't 
    leave the db in an inconsistent state """
    for i in threadList:
        i.exitAll = True

threadList = []

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
    logger.addHandler(fHandler)

    if mainConfigFile == []:
        logger.critical("Could not read main config file")
        sys.exit(1)
    if pluginConfigFiles == []:
        logger.critical("Could not read config files")
        sys.exit(1)
    logger.info("Starting topologyAnalyser daemon")
    localSession = dbmanager.initializeDB(parser)

    C = parseArgs() 

    if C.getParam("daemonMode"):
        if os.fork():
            sys.exit()
    else:
        logger.addHandler(sHandler)

    vL = C.getParam("verbosity")
    if vL == 0:
        logger.setLevel(logging.ERROR)
    elif vL == 1:
        logger.setLevel(logging.INFO)
    elif vL == 2:
        logger.setLevel(logging.DEBUG)

    # TODO automic handling of plugins based on files

    # TODO add a runlevel to initialize(). Use runlevel 0 to
    # only return a list of valid configuration parameters
    # then add them to the CL argument parsing in order to override 
    # config files

    nnx = ninux()
    nnx.initialize(parser, localSession)
    ffg = FFGraz()
    ffg.initialize(parser, localSession)
    ffw = FFWien()
    ffw.initialize(parser, localSession)
    
    if nnx.enabled:
        threadList.append(nnx)
    if ffg.enabled:
        threadList.append(ffg)
    if ffw.enabled:
        threadList.append(ffw)

    signal.signal(signal.SIGTERM, termHandler)
    # This is a throwaway variable to deal with a python bug
    throwaway = datetime.datetime.strptime('20110101','%Y%m%d')
    try:
        for i in threadList:
            i.daemon = True
            #FIXME run only the enabled ones
            i.start()
        while True:
            watchDog = 0
            for i in threadList:
                i.join(1)
                watchDog += i.exitAll
            if watchDog == len(threadList):
                break
    except KeyboardInterrupt:
        termHandler(None,None)

    
    logger.info("Received KILL signal")
    waitTime = 20
    while True:
        a = sum([i.is_alive() for i in threadList])
        if a != 0:
            logger.info("Waiting %s seconds for %d threads to gracefully exit",\
                    waitTime, a) 
        else:
            break
        if waitTime <= 0:
            logger.info("Ok, killing the threads")
            sys.exit()
        time.sleep(5)
        waitTime -= 5
    logger.info("Ok, all thread exited gracefully")


