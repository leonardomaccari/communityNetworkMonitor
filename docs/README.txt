
This is a python daemon that periodically fetches information from a few
community networks, saves historical series in a local sqlite file 
and analyses their topology features.

list of files, commented:

├── conf # configuration files, 
│   ├── accesskeys # configuration files for each network (missing in git)
│   │   ├── FFGraz.conf
│   │   └── ninux.conf
│   └── main.conf # main config
├── db # folder for the local sqlite db 
├── docs 
│   └── README.txt
└── src
    ├── dbmanager.py # libs for the local database
    ├── plugins # plugins for each community network
    │   ├── FFGraz.py # FunkFeuer Graz plugin
    │   ├── __init__.py
    │   ├── ninux.py # ninux plugin
    │   └── plugin.py # base class for plugins
    └── topologyAnalyser.py # main program file

Each network is parsed looking for nodes, links and OLSR ETX values, each scan
produces an update for the db described by the classes in dbmanager.py. 
Depending on the network the scan may be performed using direct access to the
node db, crawling the website for the available topology files or crawling the
JSON interface. See each plugin documentation in the source code of the plugin.

The main config file main.conf contains

[main]
localdb = sqlite:///../testdb/ninux.db
logfile = /tmp/community_log.txt

localdb can be changed to any sqlalchemy supported engine, (for instance
postgresql://scott:tiger@localhost:5432/mydatabase) logfile is the file where
the daemon logs its activity.

each plugin will have its own configuration parameters. Ninux plugin
configuration is not included, since it needs to connect to a db with
user and password. An example, with some parameters shared by all the
plugin follows (note that the plugin name must be the same as the config
entry): 

[FFGraz]
enable = True    # enable/disable the plugin
logLevel = INFO  # default log level DEBUG/INFO/WARN/CRITICAL/ERROR
period = 10m     # activation-period (to call the getStats() of the plugin)
                 # supports s/m/h/d

# these are plugin specific parameters

baseTopoURL = http://stats.ffgraz.net/topo/DOTARCHIV/

