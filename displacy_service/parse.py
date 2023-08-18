#import spacy
import logging
from coreferee.manager import CorefereeBroker
from spacy.tokens import Token
from spacy.language import Language

logger = None

def init_parse_logging():
    global logger
    logger = logging.getLogger("parse")
    logger.info("logging initialized")

@Language.component("merge_ners")
def merge_ners(doc):
    ents = doc.ents
    with doc.retokenize() as retokenizer:
        for ent in ents:
            retokenizer.merge(ent)
    return doc

class Parse(object):
    def __init__(self, nlp, text, collapse_punctuation, collapse_phrases):
        self.doc = nlp(text)
        with self.doc.retokenize() as retokenizer:
            if collapse_punctuation:
                spans = []
                for word in self.doc[:-1]:
                    if word.is_punct:
                        continue
                    if not word.nbor(1).is_punct:
                        continue
                    start = word.i
                    end = word.i + 1
                    while end < len(self.doc) and self.doc[end].is_punct:
                        end += 1
                    span = self.doc[start: end]
                    retokenizer.merge(self.doc[start: end])

        with self.doc.retokenize() as retokenizer:
            if collapse_phrases:
                for np in list(self.doc.noun_chunks):
                    retokenizer.merge(np)

    def to_json(self):
        words = [{'text': w.text, 'tag': w.tag_, 'pos': w.pos_, 'idx': w.idx, 'ent_type': w.ent_type_} for w in self.doc]
        arcs = []
        for word in self.doc:
            if word.i < word.head.i:
                arcs.append(
                    {
                        'start': word.i,
                        'end': word.head.i,
                        'label': word.dep_,
                        'text': str(word),
                        'dir': 'left'
                    })
            elif word.i > word.head.i:
                arcs.append(
                    {
                        'start': word.head.i,
                        'end': word.i,
                        'label': word.dep_,
                        'text': str(word),
                        'dir': 'right'
                    })
        return {'words': words, 'arcs': arcs}


class Entities(object):
    def __init__(self, nlp, text, resolve_corefs=False, unify_entities=True, collapse_phrases=False):
        self.resolve_corefs = resolve_corefs
        self.unify_entities = unify_entities
        if resolve_corefs and not collapse_phrases:
            if not nlp.has_pipe("coreferee"):
                nlp.add_pipe("coreferee")
            self.resolve_corefs = resolve_corefs
        self.doc = nlp(text)
        if collapse_phrases:
            if not Token.has_extension("original tokens"):
                Token.set_extension("original_tokens", default=None, force=True)
            with self.doc.retokenize() as retokenizer:
                    for np in list(self.doc.noun_chunks):
                        ents = np.ents
                        if len(ents)==1:
                            if np.end - np.start > 1:
                                if np.start==ents[0].start and np.end==ents[0].end:
                                    logger.debug(f"Merging {np}")
                                    retokenizer.merge(np, {"_": {"original_tokens": list([c.text for c in np])}})
                                else:
                                    if ents[0].end-ents[0].start > 1:
                                        logging.debug(f"Merging entity {ents[0]} inside noun chunk {np}")
                                        original_tokens = list([c.text for c in self.doc[ents[0].start:ents[0].end]])
                                        underscore_map = {"original_tokens": original_tokens}
                                        attributes = {"_": underscore_map}
                                        retokenizer.merge(ents[0], attributes)
                        else:
                            logger.debug(f"Not merging multi-entity noun chunk {np}")
            if resolve_corefs:
                self.resolve_corefs = resolve_corefs
                coreferee = CorefereeBroker(nlp, "")
                coreferee(self.doc)
                logger.debug(f"From Coreferee coreference chains: {self.doc._.coref_chains.pretty_representation}")
        self.unified_entities = {}
        logger.debug(f"Created Entities object with resolve_corefs={resolve_corefs}, unify_entities={unify_entities}, collapse_phrases={collapse_phrases}")

    def is_clusterable_entity(self, ent):
        type = ent.label_
        if self.resolve_corefs == True:
            return True
        elif isinstance(self.resolve_corefs, list):
            return type in self.resolve_corefs
        elif isinstance(self.resolve_corefs, str):
            return type == self.resolve_corefs
        else:
            return False

    def get_clusters(self, ent):
        clusterable = self.is_clusterable_entity(ent)
        logger.debug(f"Get clusters for {str(ent)} ({ent.label_}).  Clusterable={clusterable}")
        if clusterable == False:
            logger.debug(f"Not clustering {str(ent)} because its type {ent.label_} is disabled")
            return None
        clusters = []
        for cluster_number, chain in enumerate(self.doc._.coref_chains):
            for mention in chain:
                mentions_this_entity = next((token_index for token_index in mention
                                             if token_index >= ent.start and token_index < ent.end), None)
                if mentions_this_entity is not None:
                    logger.debug(f"coreference chains from coreferee: {chain.pretty_representation}")
                    clusters.append([[token_index for token_index in mention] for mention in chain])
        logger.debug(f"Found {len(clusters)} for entity {str(ent)}")
        return clusters if len(clusters) > 0 else None

    def make_entity_response(self, ent):
        response = {
            'start_token': ent.start,
            'end_token': ent.end,
            'start': ent.start_char,
            'end': ent.end_char,
            'type': ent.label_,
            'text': str(ent)
        }
        if self.resolve_corefs:
            clusters = self.get_clusters(ent)
            if ent in self.unified_entities:
                if clusters is not None:
                    clusters = [clusters[0] + self.unified_entities[ent]]
                else:
                    clusters = [self.unified_entities[ent]]
            if clusters is not None:
                response = response | {'clusters': clusters}
        return response

    def entity_words(self, entity):
        if Token.has_extension("original_tokens"):
            return [word for token in entity for word in (token._.original_tokens if token._.original_tokens is not None else [token.text])]
        else:
            return [token.text for token in entity]
    
    def matching_entity_type_filter(self, entity):
        return entity.label_ == "PERSON"

    ''' Having seen Jane Smith as a mult-token entity previously, presume that
        an entity with the name Jane or the name Smith refers to the same
        entity. (Jane Smith becomes the antecedant of the entities Jane and Smith.)
        Copy those entities' coreference clusters and those entities
        themselves into the clusters of Jane Smith. Return a map from each such entity
        to their antecedant.'''
    def find_matching_entities(self):
        tokens_to_entities = {}
        entity_crossreference = {}
        for entity in filter(lambda e:self.matching_entity_type_filter(e), self.doc.ents):
            if entity not in entity_crossreference:
                entity_crossreference[entity] = [entity]
            entity_text = entity.text
            entity_size = entity.end - entity.start
            for tok in entity:
                token_text = tok.text
                token_index = tok.idx
                logger.debug(f"entity token: {token_text}")
                if token_text in tokens_to_entities:
                    if all(t in tokens_to_entities for t in self.entity_words(entity)):
                        entity0 = tokens_to_entities[token_text][0]
                        entity0_size = entity0.end - entity0.start
                        if token_index > entity0.start and entity_size <= entity0_size:
                            entity_list = tokens_to_entities[token_text]
                            if entity not in entity_list:
                                entity_list.append(entity)
                                logger.debug(f"Adding {str(entity)} to entity list for token {token_text}")
                    else:
                        logger.debug(f"Dropped {token_text} because not all of {self.entity_words(entity)} in tokens_to_entities")
                else:
                    #print(f"New token? {token_text} => {self.entity_words(entity)}, size {entity.end-entity.start}")
                    for tok1 in self.entity_words(entity):
                        logger.debug(f"Adding {tok1} for entity {entity}")
                        tokens_to_entities[tok1] = entity_crossreference[entity]
        logger.info(f"Tokens to entities:: {tokens_to_entities}")
        logger.info(f"Entity cross-reference: {entity_crossreference}")
        entity_map={}
        for entity,entities in entity_crossreference.items():
            if len(entities)>1:
                for e in entities[1:]:
                    entity_map[e] = entity
        logger.info(f"Entity map: {entity_map}")
        return entity_map
        

    def make_token_response(self, token):
        return {'start': token.idx,
                'end': token.idx+len(token.text)}

    def to_json(self):
        if self.unify_entities:
            unified_entities = self.find_matching_entities()
            entities=[]
            for e in self.doc.ents:
                if e in unified_entities:
                    primary_entity = unified_entities[e]
                    clusters = self.get_clusters(e)
                    if primary_entity not in self.unified_entities:
                        self.unified_entities[primary_entity] = []
                    if e.end > e.start+1:
                        self.unified_entities[primary_entity].append([e.start, e.end-1])
                    else:
                        self.unified_entities[primary_entity].append([e.start])
                    if clusters is not None:
                        self.unified_entities[primary_entity].extend(clusters[0])
                else:
                    entities.append(e)
            entities = [self.make_entity_response(ent) for ent in entities]
        else:
            entities = [self.make_entity_response(ent) for ent in self.doc.ents]
        tokens = [self.make_token_response(token) for token in self.doc]
        return {'entities': entities,
                'tokens': tokens}


class Sentences(object):
    def __init__(self, nlp, text):
        self.doc = nlp(text)

    def to_json(self):
        sents = [sent.text.strip() for sent in self.doc.sents]
        return sents


class SentencesDependencies(object):
    def __init__(self, nlp, text, collapse_punctuation, collapse_phrases):
        '''
        if collapse_phrases is True:
            nlp.disable_pipe("merge_ners")
            nlp.disable_pipe("reparser")
        elif collapse_phrases is False:
            nlp.disable_pipe("merge_ners")
            nlp.disable_pipe("reparser")
        elif collapse_phrases == "Entities":
            nlp.enable_pipe("merge_ners")
            nlp.enable_pipe("reparser")
        '''

        self.doc = nlp(text)

        if collapse_punctuation:
            with self.doc.retokenize() as retokenizer:
                spans = []
                for word in self.doc[:-1]:
                    if word.is_punct:
                        continue
                    if not word.nbor(1).is_punct:
                        continue
                    start = word.i
                    end = word.i + 1
                    while end < len(self.doc) and self.doc[end].is_punct:
                        end += 1
                    span = self.doc[start: end]
                    retokenizer.merge(span)
                    
        if collapse_phrases is True:
            with self.doc.retokenize() as retokenizer:
                for np in list(self.doc.noun_chunks):
                    retokenizer.merge(np)
                    #np.merge(np.root.tag_, np.root.lemma_, np.root.ent_type_)

    def to_json(self):
        sents = []
        for sent in self.doc.sents:
            words = [{**{'text': w.text, 'tag': w.tag_, 'pos': w.pos_, 'idx': w.idx, 't_idx': w.i}, **({'lemma': w.lemma_} if w.lemma_ != w.text else {})} for w in sent]
            arcs = []
            for word in sent:
                if word.i < word.head.i:
                    arcs.append(
                        {
                            'start': word.i,
                            'end': word.head.i,
                            'label': word.dep_,
                            'text': str(word),
                            'dir': 'left'
                        })
                elif word.i > word.head.i:
                    arcs.append(
                        {
                            'start': word.head.i,
                            'end': word.i,
                            'label': word.dep_,
                            'text': str(word),
                            'dir': 'right'
                        })

            sents.append({'sentence': sent.text.strip(),
                          'dep_parse': {'words': words,
                                        'arcs': arcs}})
        return sents
