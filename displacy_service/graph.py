import falcon
import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
import logging
import json
import os
import re
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
    logger = logging.getLogger("SubgraphIsomorphismResource")
    logger.debug("subgraph isomorphism resource logging initialized")

    def __init__(self):
        SubgraphIsomorphismResource.init_subgraphs()

    @staticmethod
    def init_subgraphs():
        if (len(SubgraphIsomorphismResource.subgraphs) == 0 or
            os.stat("subgraphs.json").st_mtime_ns > SubgraphIsomorphismResource.subgraph_fingerprint):
            SubgraphIsomorphismResource.logger.debug("Reading subgraphs.json")
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
        SubgraphIsomorphismResource.logger.debug(f"Node match {n1} === {n2} => {match}")
        if match:
            if n2.get("text") is not None:
                if isinstance(n2.get("text"),str):
                    match = n1.get("text") == n2.get("text")
                else:
                    match = n1.get("text") in n2.get("text")
                SubgraphIsomorphismResource.logger.debug(f"Template has text constraint {n2.get('text')}.  After text match, node match {n1} === {n2} => {match}")
        return match

    @staticmethod
    def edge_matcher(e1, e2):
        match = e1["label"]==e2["label"]
        SubgraphIsomorphismResource.logger.debug(f"Edge match {e1} === {e2} => {match}")
        return match

    @staticmethod
    def subtree_to_text(graph, start_node, use=["text"], stop_at=None, stop_after=None):
        left_labels = ["amod", "nmod", "compound"]
        def stop_predicate(nbr_edge):
            nbr = nbr_edge[0]
            edge = nbr_edge[1]
            if isinstance(stop_at,list):
                return edge["label"] in stop_at
            elif isinstance(stop_at,int):
                return nbr > stop_at
            else:
              return False
        SubgraphIsomorphismResource.logger.debug(f"subtree_to_text: start_node={start_node}")
        outs = graph.adj[start_node]
        SubgraphIsomorphismResource.logger.debug(f"adj: {outs}, node={graph[start_node]}")
        node_text = next(graph.nodes[start_node][field] for field in use if field in graph.nodes[start_node])
        if len(outs)==0:
            return node_text
        else:
            before_nodes = list([nbr for nbr, edge in outs.items() if nbr < start_node and edge["label"] in left_labels])
            after_nodes = takewhile(lambda nbr_edge: not(stop_predicate(nbr_edge)),
                                    [(nbr,edge) for nbr, edge in outs.items() if nbr > start_node])
            befores = " ".join([graph.nodes[nbr]["text"] for nbr in before_nodes])
            afters = " ".join([SubgraphIsomorphismResource.subtree_to_text(graph, nbr, use=use, stop_at=stop_at)
                               for nbr, edge in after_nodes])
            return befores + " " + node_text + " " + afters
        
    @staticmethod
    def instantiate_template(template, use, graph, match):
        def replacer(m):
            template_node_index = int(m.group(1))
            graph_node_index = match[template_node_index]
            graph_node = graph.nodes[graph_node_index]
            replacements = [graph_node.get(field) for field in use]
            replacement = next(filter(lambda v: v is not None, replacements))
            SubgraphIsomorphismResource.logger.debug(f"{template}: {template_node_index} => {graph_node_index} => {graph_node}")
            if replacement is not None:
                return replacement
            else:
                return f"${template_node_index}"
            
        instantiation= re.sub(r'\$(\d+)', replacer, template)
        SubgraphIsomorphismResource.logger.debug(f"Instantiate template {template} with match {match} using {use} => {instantiation}")
        return instantiation
        
    @staticmethod
    def get_rdf_value(rdf_part, match, graph):
        if "value" in rdf_part:
            return rdf_part["value"]
        elif "ref" in rdf_part or "template" in rdf_part:
            if rdf_part.get("template") is not None:
                value = SubgraphIsomorphismResource.instantiate_template(rdf_part.get("template"),
                                                                         rdf_part.get("use", ["text"]),
                                                                         graph, match)
            else:
                subgraph_node = rdf_part["ref"]
                graph_node = match[subgraph_node]
                stop_at = rdf_part.get("stop_at", None)
                if isinstance(stop_at, int):
                    stop_at = stop_at - subgraph_node + graph_node + 1
                value = SubgraphIsomorphismResource.subtree_to_text(graph, graph_node,
                                                                    use=rdf_part.get("use", ["text"]),
                                                                    stop_at=stop_at)
                if stop_at is not None:
                    SubgraphIsomorphismResource.logger.debug(f"get_rdf_value: {subgraph_node}-{rdf_part.get('stop_at')} => {graph_node}-{stop_at}: {value}")
                else:
                    SubgraphIsomorphismResource.logger.debug(f"get_rdf_value: {subgraph_node}-* => {graph_node}-*: {value}")
            return value
        
    @staticmethod
    def get_rdf(match, subgraph, graph):
        SubgraphIsomorphismResource.logger.debug(f"get_rdf: match: {match}")
        rdf = subgraph.graph["rdf"]
        imatch = {val:key for key,val in match.items()}
        return [{key: SubgraphIsomorphismResource.get_rdf_value(val, imatch, graph)  for key,val in onerdf.items()}
                for onerdf in rdf]
    
    @staticmethod
    def match_graph(subgraph, graph):
        SubgraphIsomorphismResource.logger.setLevel(subgraph.graph.get('logging', logging.INFO))
        SubgraphIsomorphismResource.logger.debug(f"Trying {subgraph.graph['pattern']}")
        matcher = DiGraphMatcher(graph, subgraph,
                                 node_match=SubgraphIsomorphismResource.node_matcher,
                                 edge_match=SubgraphIsomorphismResource.edge_matcher)
        if matcher.subgraph_is_isomorphic():
            mapping = list([rdf for m in matcher.subgraph_isomorphisms_iter() for rdf in SubgraphIsomorphismResource.get_rdf(m, subgraph, graph) ])
            SubgraphIsomorphismResource.logger.debug(f"{subgraph.graph['pattern']} found {len(mapping)} matches.")
            return {"pattern": subgraph.graph["pattern"],
                    "rdfs": mapping}
        else:
            SubgraphIsomorphismResource.logger.debug(f"{subgraph.graph['pattern']} found 0 matches.")
            return None
        
    def on_post(self, req, resp):
        SubgraphIsomorphismResource.logger.debug("Matching graphs")
        SubgraphIsomorphismResource.init_subgraphs()
        req_body = req.bounded_stream.read()
        graph_data = json.loads(req_body.decode('utf8'))

        graph = SubgraphIsomorphismResource.make_graph(graph_data)
        result = [SubgraphIsomorphismResource.match_graph(subgraph, graph) for subgraph in
                  SubgraphIsomorphismResource.subgraphs]

        resp.content_type = "application/json"
        resp.status = falcon.HTTP_200
        resp.body = json.dumps(list(filter(lambda r: r is not None, result)))
        
