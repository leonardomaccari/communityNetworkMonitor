
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
    │   ├── FFWien.py # FunkFeuer Wien plugin
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


========== Plugins documentation

== FFWien

This plugin downloads the topology of FFWien from a base url where the
OLSR dump is saved in stats.funkfeuer.at. Since I want to aggregate
links that belong to the same physical installation (~wired) I also get
the association between a physical node and all its IPs from the JSON
interface. This second operation takes a long time, so I do it with a
different period. The info extracted from the JSON is saved in the local
DB. Roughly, the procedure is as follows:

1) get list of wireless interfaces from
https://ff-nodedb.funkfeuer.at//api/FFM-Wireless_Interface
2) get each interface, get "attributes":pid (IFACEPID).
for each nterface get ipv4 network links from
/api/FFM-Wireless_Interface/**IFACEID**/ip4_network_links"
and the field "attributes":"left":"url"  

3a) For each network_link get the list of all the entries:
"entries": [
"/api/FFM-Wireless_Interface_in_IP4_Network/2568",
...
]
4a) for each entry, get the entry and get from the entry:
"attributes":"right":"url" 
5a) from the IP4_network get
"attributes": "net address" (IP address)

3b) for the net-device get "attributes":"node":pid
 
This procedure stores enough information in the DB to be able to build a
map IP address -> Node ID

4) make a graph with the OLSR map and substitute each IP with 
   the corresponding nodepid. (CHECK that each IP gets a node)

== FFGraz

Here I get a .dot file that is published at a specified url every 10
minutes. The naming scheme adopted by FFGraz is interface.device.node,
so the .dot file is of the kind:

if0.dev0.node0 -> if1.dev0.node1

The plugin parses the names and aggregates interfaces/devices belonging
to the same node.
