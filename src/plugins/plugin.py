from dbmanager import *
import logging
import os

class plugin():
    parser = None
    localSession = None
    logger = None

    def baseInitialize(self, parser, lc, name):
        self.logger= logging.getLogger("plugin base")
        self.logger.setLevel(logging.INFO)
        pluginName = os.path.splitext(os.path.basename(name))[0]
        import code
        code.interact(local=locals())
        if pluginName not in parser.sections():
            self.logger.error("Could not load config file for %s network,\
                    exiting", pluginName)
        try:
            logLevel = self.parser.get(pluginName, 'loglevel')
            if logLevel == "DEBUG":
                returnLevel = logging.DEBUG
            if logLevel == "INFO":
                returnLevel = logging.INFO
            if logLevel == "WARNING":
                returnLevel = logging.WARNING
            if logLevel == "ERROR":
                returnLevel = logging.ERROR
            if logLevel == "CRITICAL":
                returnLevel = logging.CRITICAL
            else:
                returnLevel = logging.INFO
        except:
            returnLevel = logging.INFO
        return returnLevel, pluginName
