
DB=$1
NETWORK=$2

QUERY="select snode.Id, dnode.Id, etx.etx_value from \
  link, scan, node as snode, node as dnode, etx \
  WHERE link.scan_Id = scan.Id AND snode.Id = link.from_node_Id \
  AND dnode.Id = link.to_node_Id AND etx.link_Id = link.Id \
  AND dnode.scan_Id = scan.Id AND snode.scan_Id = scan.Id AND scan.Id IN \
  (SELECT Id FROM scan WHERE network=\"$NETWORK\" ORDER BY time DESC LIMIT 1);"

echo $QUERY

echo $QUERY | sqlite3 $1 
