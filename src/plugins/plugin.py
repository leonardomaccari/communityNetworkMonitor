from dbmanager import *
import logging
import os
import sys
import time
from threading import Thread

class plugin(Thread):
    disabledMessage = "Plugin disabled, doing nothing"
    exitAll = False
    session = None

    def __init__(self):
        print >> sys.stderr, "Error, you should not call init in the plugin class"
        sys.exit(1)
    
    def baseInitialize(self, parser,  name):
        """ initialize some shared parameters, do some error check """
        self.logger = logging.getLogger("plugin base")
        self.logger.setLevel(logging.INFO)
        pluginName = os.path.splitext(os.path.basename(name))[0]
        if pluginName not in parser.sections():
            self.logger.error("Could not load config file for %s network,\
                    exiting", pluginName)
        enabled = True
        try:
            en = parser.get(pluginName, 'enable')
            if en == 'False':
                enabled = False
        except:
            pass

        try:
            logLevel = parser.get(pluginName, 'loglevel')
            if logLevel == "DEBUG":
                returnLevel = logging.DEBUG
            elif logLevel == "INFO":
                returnLevel = logging.INFO
            elif logLevel == "WARNING":
                returnLevel = logging.WARNING
            elif logLevel == "ERROR":
                returnLevel = logging.ERROR
            elif logLevel == "CRITICAL":
                returnLevel = logging.CRITICAL
            else:
                self.logger.error("logLevel from config file unknown,"+\
                        "switching to INFO")
                returnLevel = logging.INFO
        except:
            returnLevel = logging.INFO
        return enabled, returnLevel, pluginName

    def convertTime(self, s):
        """ simply convert a string of the format 1s/m/h/d to the equivalent
        in seconds """

        unit = s[-1]
        value = int(s[0:-1])
        if unit == 's':
            return value
        if unit == 'm':
            return 60*value
        if unit == 'h':
            return 60*60*value
        if unit == 'd':
            return 60*60*24*value
        raise

    def run(self):
        #self.localSession = self.localSession()
        while not self.exitAll:
            self.getStats()
            for i in range(self.period):
                time.sleep(1)
                if self.exitAll:
                    break
