
This is a python daemon that periodically fetches information from a few
community networks, saves historical series in a local sqlite file 
and analyses their topology features.

Copyright is of Leonardo Maccari (leonardo.maccari@unitn.it) and the
code is released under GPLv3 license.

================ Install & Usage

You will need to install some python modules:
 - requests
 - sqlalchemy
 - networkx
 - mechanize
 - beautifulsoup
 - simplejson
 - python-mysqldb (for ninux plugin, or if you want to use any other DB
   that is not sqlite)
 - pycrypto (for anonymization)


run ./src/topologyAnalyser.py, if you have other errors for missing modules
you may need to install more python packages.

Once you satisfied the dependencies, look into the conf/main.conf file
and choose your local database type. sqlite is the easiest option (see
below for details).

The few command line parameters are self-explicative, run with -h for a
list.  Once started, the program will run until you kill it and will
save the information it fetches in the local database. For each scan you
will have a new entry in the "scan" table, a new set of nodes, links and
of etx values in the respective tables. Then you can get the information
you need with SQL. For instance, if you want to have a table with
"nodeId, nodeId, etx" for each link, for the FFWien network just for the
last scan, you can use a query like:



SELECT snode.Id, dnode.Id, etx.etx_value FROM link, scan, node AS snode,
node AS dnode, etx WHERE link.scan_Id = scan.Id AND snode.Id =
link.from_node_Id AND dnode.Id = link.to_node_Id AND etx.link_Id =
link.Id AND dnode.scan_Id = scan.Id AND snode.scan_Id = scan.Id AND
scan.Id IN (SELECT Id FROM scan WHERE network="FFWien" ORDER BY time
DESC LIMIT 1);

===================== Configurations & internals

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
    ├── topologyAnalyser.py # main program file
    └── myCrypto.py # a simple module for the encryption of data

Each network is parsed looking for nodes, links and OLSR ETX values,
each scan produces an update for the db described by the classes in
dbmanager.py.  Depending on the network the scan may be performed using
direct access to the node db, crawling the website for the available
topology files or crawling the JSON interface. See each plugin
documentation in the source code of the plugin.  Data can be anonymized.
Since i have access to non public DB of nodes, owners and email, the
daemon can be configured to encrypt with AES node names, owner names and
emails. 

The main config file main.conf contains

[main]
localdb = sqlite:///../testdb/ninux.db
logfile = /tmp/community_log.txt

localdb can be changed to any sqlalchemy supported engine, (for instance
postgresql://scott:tiger@localhost:5432/mydatabase) logfile is the file where
the daemon logs its activity.

any .conf file in the conf/ folder will be parsed. Encryption is set up
with a stanza like the following

[encrypt]
key = yourencryptionkeyhere

the key will be used to encrypt data with AES.  Note that if you have
access to a db with encrypted entries, but you don't know the keys, you
can still use it, all the encrypted entries are base64 encoded so you
can query them in the db. You will not know the names of nodes, emails
and people.

Each plugin will have its own configuration parameters. Ninux plugin
configuration is not included, since it needs to connect to a db with
user and password. Also people of FFWien, after a few weeks decided that
the access to their json db has been limited to only authenticated
people. FFWien can be accessed without credentials.
An example, with some parameters shared by all the plugin follows (note
that the plugin name must be the same as the config entry): 

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



============ FAQs ================

1) I get this error:

 sqlalchemy.exc.OperationalError: (OperationalError) unable to open
 database file None None

you probably did not set the localdb variable in the main.conf file to a
valid (existent) place

2) I get thousands of these messages:

requests.packages.urllib3.connectionpool: INFO, Starting new HTTPS connection (1): ff-nodedb.funkfeuer.at

the requests module is used in the FFWien plugin and by default logs
everything it does, you have to disable it. The only way I found is with
the lines in getStats() that do:

  reqL = logging.getLogger("urllib3.connectionpool")
  reqL.setLevel(logging.ERROR)

if you see the messages it means that your systems names the module in a
different way, you just need to modify urllib3.connectionpool with what
your system uses, that is in the first part of the message


================= known bugs

When exiting, in some systems you get an error (should have no
consequences):

Exception in thread Thread-3 (most likely raised during interpreter
shutdown):
Traceback (most recent call last):
  File "/usr/lib/python2.6/threading.py", line 532, in __bootstrap_inner
  File "/home/maccari/src/cn/src/plugins/plugin.py", line 94, in run
  File "/home/maccari/src/cn/src/plugins/FFWien.py", line 438, in
getStats
  File "/home/maccari/src/cn/src/plugins/FFWien.py", line 130, in
getJSONDump
  File "/home/maccari/src/cn/src/plugins/FFWien.py", line 207, in
getURLWrapper
  File "/usr/local/lib/python2.6/dist-packages/requests/api.py", line
55, in get
  File "/usr/local/lib/python2.6/dist-packages/requests/api.py", line
44, in request
  File "/usr/local/lib/python2.6/dist-packages/requests/sessions.py",
line 361, in request
  File "/usr/local/lib/python2.6/dist-packages/requests/sessions.py",
line 464, in send
  File "/usr/local/lib/python2.6/dist-packages/requests/adapters.py",
line 352, in send
<type 'exceptions.AttributeError'>: 'NoneType' object has no attribute
'error'

