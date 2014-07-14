from dbmanager import *
import logging
import os
import sys
import time
import simplejson
from threading import Thread
from ConfigParser import NoOptionError, NoOptionError
from myCrypto import myCrypto

class plugin(Thread):
    disabledMessage = "Plugin disabled, doing nothing"
    exitAll = False
    session = None
    aes = None

    def __init__(self):
        print >> sys.stderr, "Error, you should not call ",\
                "init in the plugin class"
        sys.exit(1)
    
    def baseInitialize(self, parser,  name, lc):
        """ initialize some shared parameters, do some error check """

        # REMINDER: do not access the db in this function, each plugin is a 
        # thread with a localSession that is separated when they become
        # a thread. Here there is no thread so the scoped_session does
        # not differentiate between sessions. And havoc occurs.
        self.logger = logging.getLogger("plugin base")
        pluginName = os.path.splitext(os.path.basename(name))[0]
        if pluginName not in parser.sections():
            self.logger.error("Could not load config file for %s network,\
                    exiting", pluginName)
        enabled = True
        try:
            en = parser.get(pluginName, 'enable')
            if en == 'False':
                enabled = False
                self.logger.info("Plugin %s disabled", pluginName)
        except:
            pass

        returnLevel = logging.getLogger().level
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
                        "switching to default")
        except:
            pass

        try:
            pseudonymFileName = parser.get(pluginName, 'pseudonymfile')
        except NoOptionError: #FIXME all the try should match this!!
            # this code takes the input file, which is expected to be a 
            # json (possibly saved with simplejson.dump() where 
            # a map of the kind {"FromStringX":"FromString"} is saved
            # each FromStringX is mapped into the FromString. 
            # You can generate the json with the option 'dumpnames' but
            # you have to process it to merge multiple pseudonyms for the 
            # same person. See README for how to do this
            # FIXME write readme
            pseudonymFileName = ""
            pass
        if pseudonymFileName != "":
            try:
                self.pseudonymFile = open(pseudonymFileName, "r")
            except IOError:
                self.logger.error("Could not open file specified",\
                        "in pseudonymfile option.")
                sys.exit(1)

        if self.pseudonymFile != None:
            try:
                self.ownerPseudonymDict = simplejson.load(self.pseudonymFile)
            except simplejson.JSONDecodeError:
                sys.logger.error("The file specified in the pseudonymfile"+\
                        "option is a malformed JSON!")
                sys.exit(1)
        try:
            pseudonymDumpFileName = parser.get(pluginName, 'pseudonymdump')
        except NoOptionError: 
            pseudonymDumpFileName = ""
        if pseudonymDumpFileName != "":
            try:
                self.pseudonymDumpFile = open(pseudonymDumpFileName, "w")
            except IOError:
                self.logger.error("Could not open file specified"+\
                        "in pseudonymdump option:"+pseudonymDumpFileName)
                sys.exit(1)

        key = ""
        #TODO this must be moved in a class for the database connection. 
        #dbmanager should become a class and the key should be moved there
        #or else, it is repeated for every plugin
        try:
            key = parser.get('encrypt', 'key')
        except NoSectionError:
            print "ERROR: you always have to specify a key for local encryption of "
            print "the database. The skey is used to anonymise node names, owners, emails"
            print "You have to add a encrypt.conf file in the conf/ folder with a [encrypt]"
            print "section and a key configuration, like:\n"
            print "[encrypt]"
            print "key = randomseedusedasinputforcrypto"
            print "If you do not want to encrypt, use an empty [encrypt] stanza, key"
            print "is hashed and truncated to 32 bytes"
            sys.exit(1)
        except NoOptionError:
            key = ""
        if key != "":
            self.myCrypto = myCrypto(key)
        return enabled, returnLevel, pluginName

    def dumpPseudonym(self, d):
        """ we dump the pseudonym file, only once """
        if self.pseudonymDumpFile != None and not \
                self.pseudonymDumpFile.closed:
            simplejson.dump(d, self.pseudonymDumpFile)
            self.pseudonymDumpFile.close()

    def addNetwork(self):
        if self.enabled:
            networkRow = self.localSession.query(network).\
                    filter_by(name=self.pluginName).first()
            if networkRow == None:
                try:
                    d = self.parser.get(self.pluginName, 'description')
                except:
                    d = ''
                newNetwork = network(name=self.pluginName, description=d)
                self.localSession.add(newNetwork)
                self.logger.info("Adding new network %s to the DB",
                        self.pluginName)


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
        self.addNetwork()
        while not self.exitAll:
            lastRun = datetime.now()
            self.getStats()
            runLenght = (datetime.now() - lastRun).seconds
            diffTime = (self.period - runLenght)
            if diffTime > 0:
                for i in range(diffTime):
                    time.sleep(1)
                    if self.exitAll:
                        break
        if self.pseudonymFile != None and not self.pseudonymFile.closed:
            self.pseudonymFile.close()
        if self.pseudonymDumpFile != None and not \
                self.pseudonymDumpFile.closed:
            self.pseudonymDumpFile.close()
