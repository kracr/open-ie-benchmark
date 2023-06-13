import os
import pandas as pd
import json
import stanza
import nltk

# os.mkdir(r"../data/graphs")

pipe = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency',  download_method=stanza.DownloadMethod.REUSE_RESOURCES)

sentence_to_graph = {}
POS_to_id = {}


def import_sentences(path):
  tree_objects = []
  sentences = []

  with open(path, 'r',  encoding="utf8") as fp:
    count = 0
    ctr = 0

    for line in list(fp.readlines()):

      output = pipe(line)
      current_sentence_tree = []
      current_sentence = []

      for parsed_line in output.sentences:
        tree = nltk.Tree.fromstring(str(parsed_line.constituency))
        current_sentence_tree.append(tree)
        current_sentence.append(parsed_line.text)
        sentence_to_graph[count] = parsed_line.text
      
      sentences.append(current_sentence)
      tree_objects.append(current_sentence_tree)

      
  
  return tree_objects, sentences

tree_objects, sentences = import_sentences("./data/webnlg.txt")

with open('./sentences.txt', 'w') as dp:
   for i in sentences:
    dp.writelines(''.join(i)+'\n')


class Graph:
  def __init__(self, trees, MAX_DEPTH=4):
    self.trees = trees
    self.sentence_to_graph = []
    self.MAX_DEPTH = MAX_DEPTH

  def make_graph(self):
    for tree in self.trees:
      self.temp = []
      for t in tree:
        self.traverse_tree(t, 0, None)
      
      self.sentence_to_graph.append(self.temp)

  def traverse_tree(self, tree, depth, predecessor):

    if depth==self.MAX_DEPTH:
      return None

    for subtree in tree:
      if type(subtree) == str:
        continue
      
      current_node = subtree.label()
      
      if predecessor is not None:
        self.temp.append((predecessor, current_node))

      if type(subtree) == nltk.tree.Tree:
        self.traverse_tree(subtree, depth+1, current_node)


gg = Graph(trees=tree_objects)
gg.make_graph()
pos = 0
count = 0

for gr in gg.sentence_to_graph:

    src = './data/raw/graph_' + str(count) + ".txt"
    count = count + 1
    with open(src, 'w') as op:
        for edges in gr:
            if POS_to_id.get(edges[0], -1) == -1:
                POS_to_id[edges[0]] = pos
                pos = pos+1
            
            if POS_to_id.get(edges[1], -1) == -1:
                POS_to_id[edges[1]] = pos
                pos = pos+1

            op.write(f"{POS_to_id[edges[0]]} {POS_to_id[edges[1]]}\n")


with open('./POS_TO_ID.json', 'w') as pid:
   json.dump(POS_to_id, pid)

with open('./sentence_to_graph.json', 'w') as sid:
   json.dump(sentence_to_graph, sid)
