from tqdm import tqdm
import wordnet
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec


graph = nx.Graph()
lex_funs = {}
for synset in tqdm(wordnet.all_synsets(), desc="Building graph"):
    for rel in (synset.hypernyms()+
                synset.instance_hypernyms()+
                synset.hyponyms()+
                synset.instance_hyponyms()+
                synset.member_holonyms()+
                synset.substance_holonyms()+
                synset.part_holonyms()+
                synset.member_meronyms()+
                synset.substance_meronyms()+
                synset.part_meronyms()):
        graph.add_edge(synset.synsetOffset, rel.synsetOffset)
    for lexeme in synset.lexemes():
        lex_funs[lexeme.lex_fun] = synset.gloss
        graph.add_edge(synset.synsetOffset, lexeme.lex_fun)
        for rel in (lexeme.antonyms()+
                    lexeme.participle()+
                    lexeme.alsosee()+
                    lexeme.derived()):
            graph.add_edge(lexeme.lex_fun, rel.lex_fun)

node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=8)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
model.save("word2vec.model")
    
with open("vectors.tsv","w") as vf, open("meta.tsv","w") as mf:
    mf.write("function\tgloss\n")
    for lex_fun, gloss in lex_funs.items():
        vf.write("\t".join(map(str,model.wv[lex_fun]))+"\n")
        mf.write(lex_fun+"\t"+gloss+"\n")
