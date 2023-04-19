import falcon
import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
import logging
import json

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
    @staticmethod
    def node_matcher(n1, n2):
        logging.info(f"{n1} === {n2}")
        if "pos" in n1 and "pos" in n2 and "tag" in n1 and "tag" in n2:
            return n1["pos"] == n2["pos"]
        elif "pos" in n1 and "pos" in n2:
            return n1["pos"] == n2["pos"]
        elif "tag" in n1 and "tag" in n2:
            return n1["tag"] == n2["tag"]
        else:
            return False

    @staticmethod
    def edge_matcher(e1, e2):
        logging.info(f"{e1} === {e2}")
        return e1["label"]==e2["label"]

    def on_post(self, req, resp):
        logging.debug("Matching graphs")
        req_body = req.bounded_stream.read()
        json_data = json.loads(req_body.decode('utf8'))
        graph_data = json_data["graph"]
        subgraph_data = json_data["subgraph"]

        graph = nx.DiGraph()
        edges = graph_data["arcs"]
        nodes = graph_data["words"]
        for node_index, node in enumerate(nodes):
            graph.add_node(node_index, pos=node["pos"], tag=node["tag"], text=node["text"])
        for edge in edges:
            if edge["dir"] == "left":
                graph.add_edge(edge["end"], edge["start"], label=edge["label"])
            else:
                graph.add_edge(edge["start"], edge["end"], label=edge["label"])

        subgraph = nx.DiGraph()
        edges = subgraph_data["arcs"]
        nodes = subgraph_data["words"]
        for node_index, node in enumerate(nodes):
            subgraph.add_node(node_index, pos=node.get("pos", None), tag=node.get("tag", None))
        for edge in edges:
            if edge["dir"] == "left":
                subgraph.add_edge(edge["end"], edge["start"], label=edge["label"])
            else:
                subgraph.add_edge(edge["start"], edge["end"], label=edge["label"])

        matcher = DiGraphMatcher(graph, subgraph,
                                 node_match=SubgraphIsomorphismResource.node_matcher,
                                 edge_match=SubgraphIsomorphismResource.edge_matcher)

        resp.content_type = "application/json"
        resp.status = falcon.HTTP_200
        if matcher.subgraph_is_isomorphic():
            mapping = matcher.mapping
            resp.body = json.dumps(mapping)
        else:
            resp.body = "null"

