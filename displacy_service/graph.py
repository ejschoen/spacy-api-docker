import falcon
import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
import logging
import json
import os
from itertools import takewhile

logger = None

def init_graph_logging():
    global logger
    logger = logging.getLogger("graph")
    logger.info("graph logging initialized")

class GraphTestResource(object):
    """Load a spacy dependency parse graph"""
    def on_post(self, req, resp):
        req_body = req.bounded_stream.read()
        json_data = json.loads(req_body.decode('utf8'))
        for parse in json_data:
            graph = nx.DiGraph()
            edges = parse["dep_parse"]["arcs"]
            nodes = parse["dep_parse"]["words"]
            for node_index,node in enumerate(nodes):
                graph.add_node(node_index, pos=node["pos"], tag=node["tag"], text=node["text"])
            for edge in edges:
                if edge["dir"] == "left":
                    graph.add_edge(edge["end"],edge["start"], label=edge["label"])
                else:
                    graph.add_edge(edge["start"],edge["end"], label=edge["label"])
        resp.body="OK"
        resp.content_type="text/string"
        resp.status = falcon.HTTP_200

class SubgraphIsomorphismResource(object):
    subgraphs = []
    subgraph_fingerprint = None
    
    def __init__(self):
        SubgraphIsomorphismResource.init_subgraphs()

    @staticmethod
    def init_subgraphs():
        if (len(SubgraphIsomorphismResource.subgraphs) == 0 or
            os.stat("subgraphs.json").st_mtime_ns > SubgraphIsomorphismResource.subgraph_fingerprint):
            logging.info("Reading subgraphs.json")
            with open("subgraphs.json", "r") as f:
               SubgraphIsomorphismResource.subgraph_fingerprint = os.stat("subgraphs.json").st_mtime_ns
               SubgraphIsomorphismResource.subgraphs = [SubgraphIsomorphismResource.make_graph(data) for data in json.load(f)]
        
    @staticmethod
    def make_graph(data):
        graph = nx.DiGraph(rdf=data.get("rdf", None), pattern=data.get("pattern", None))
        edges = data["arcs"]
        nodes = data["words"]
        node_offset = 0
        if len(nodes)>0 and "t_idx" in nodes[0]:
            node_offset = nodes[0]["t_idx"]
        for node_index, node in enumerate(nodes):
            graph.add_node(node_index, **node)
        for edge in edges:
            if edge["dir"] == "left":
                graph.add_edge(edge["end"]-node_offset, edge["start"]-node_offset, label=edge["label"])
            else:
                graph.add_edge(edge["start"]-node_offset, edge["end"]-node_offset, label=edge["label"])
        return(graph)

    @staticmethod
    def node_matcher(n1, n2):
        match = False
        if "pos" in n1 and "pos" in n2 and "tag" in n1 and "tag" in n2:
            pos1 = n1["pos"]
            pos2 = n2["pos"]
            if isinstance(pos2,str):
                match = pos1 == pos2
            else:
                match = pos1 in pos2
        elif "pos" in n1 and "pos" in n2:
            pos1 = n1["pos"]
            pos2 = n2["pos"]
            if isinstance(pos2,str):
                match = pos1 == pos2
            else:
                match = pos1 in pos2
        elif "tag" in n1 and "tag" in n2:
            tag1 = n1["tag"]
            tag2 = n2["tag"]
            if isinstance(tag2, str):
                match = tag1 == tag2
            else:
                match = tag1 in tag2
        logging.debug(f"Node match {n1} === {n2} => {match}")
        return match

    @staticmethod
    def edge_matcher(e1, e2):
        match = e1["label"]==e2["label"]
        logging.debug(f"Edge match {e1} === {e2} => {match}")
        return match

    @staticmethod
    def subtree_to_text(graph, start_node, use=["text"], stop_at=None):
        def stop_predicate(nbr_edge):
            nbr = nbr_edge[0]
            edge = nbr_edge[1]
            if isinstance(stop_at,list):
                return edge["label"] in stop_at
            elif isinstance(stop_at,int):
                return nbr > stop_at
            else:
              return False
        logging.info(f"subtree_to_text: start_node={start_node}")
        outs = graph.adj[start_node]
        logging.info(f"adj: {outs}, node={graph[start_node]}")
        node_text = next(graph.nodes[start_node][field] for field in use if field in graph.nodes[start_node])
        if len(outs)==0:
            return node_text
        else:
            neighbors = takewhile(lambda nbr_edge: not(stop_predicate(nbr_edge)),
                                  [(nbr,edge) for nbr, edge in outs.items() if nbr > start_node])
            afters = " ".join([SubgraphIsomorphismResource.subtree_to_text(graph, nbr, use=use, stop_at=stop_at)
                               for nbr, edge in neighbors])
            return node_text + " " + afters
        
    @staticmethod
    def get_rdf_value(rdf_part, match, graph):
        if "value" in rdf_part:
            return rdf_part["value"]
        elif "ref" in rdf_part:
            subgraph_node = rdf_part["ref"]
            graph_node = match[subgraph_node]
            stop_at = rdf_part.get("stop_at", None)
            if isinstance(stop_at, int):
                stop_at = stop_at - subgraph_node + graph_node + 1
            value = SubgraphIsomorphismResource.subtree_to_text(graph, graph_node,
                                                                use=rdf_part.get("use", ["text"]),
                                                                stop_at=stop_at)
            if stop_at is not None:
                logging.info(f"get_rdf_value: {subgraph_node}-{rdf_part.get('stop_at')} => {graph_node}-{stop_at}: {value}")
            else:
                logging.info(f"get_rdf_value: {subgraph_node}-* => {graph_node}-*: {value}")
            return value
        
    @staticmethod
    def get_rdf(match, subgraph, graph):
        logging.info(f"get_rdf: match: {match}")
        rdf = subgraph.graph["rdf"]
        imatch = {val:key for key,val in match.items()}
        return [{key: SubgraphIsomorphismResource.get_rdf_value(val, imatch, graph)  for key,val in onerdf.items()}
                for onerdf in rdf]
    
    @staticmethod
    def match_graph(subgraph, graph):
        logging.info(f"Trying {subgraph.graph['pattern']}")
        matcher = DiGraphMatcher(graph, subgraph,
                                 node_match=SubgraphIsomorphismResource.node_matcher,
                                 edge_match=SubgraphIsomorphismResource.edge_matcher)
        if matcher.subgraph_is_isomorphic():
            mapping = list([rdf for m in matcher.subgraph_isomorphisms_iter() for rdf in SubgraphIsomorphismResource.get_rdf(m, subgraph, graph) ])
            return {"pattern": subgraph.graph["pattern"],
                    "rdfs": mapping}
        else:
            return None
        
    def on_post(self, req, resp):
        logging.debug("Matching graphs")
        req_body = req.bounded_stream.read()
        graph_data = json.loads(req_body.decode('utf8'))

        graph = SubgraphIsomorphismResource.make_graph(graph_data)
        result = [SubgraphIsomorphismResource.match_graph(subgraph, graph) for subgraph in
                  SubgraphIsomorphismResource.subgraphs]

        resp.content_type = "application/json"
        resp.status = falcon.HTTP_200
        resp.body = json.dumps(list(filter(lambda r: r is not None, result)))
        