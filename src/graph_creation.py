import json
import os
from transformers import T5Tokenizer
import dgl
import torch
from collections import deque
from tqdm import tqdm, trange


class GraphBuilder():
    def __init__(self, model_weights="t5-small", max_length=512):
        self.tokenizer = T5Tokenizer.from_pretrained(model_weights, model_max_length=max_length)
        self.max_length = max_length
        self.add_html_tags2tokenizer(self.tokenizer)
        self.id_slash_n = self.tokenizer.get_added_vocab()['\\n']
        self.id_pad = self.tokenizer.get_vocab()['<pad>']
        self.id_sep = self.tokenizer.get_vocab()['</s>']
        
    def create_graph(self, list_ctxs):
        '''
            - list_ctxs: [{'title': ..., 'text': ...}, ...]
        '''
        nodes = {'h1': 0, 'h2': 0, 'h3': 0, 'h4': 0, 'p': 0, 'li': 0, 'tr': 0, 'token': 0, 'q': 0}
        edges = {}
        
        list_input_ids = []
        list_attention_masks = []
        # prev_root_node = None
        stack = deque()
        for sec_idx, section in enumerate(list_ctxs):
            root_node = self.current_node(section['title'])
            sent_idx2node = {0: {"node_type": "q", "node_idx": nodes["q"]},
                             1: {"node_type": root_node, "node_idx": nodes[root_node]}
                            }
            nodes[root_node] += 1
            nodes["q"] += 1
            
            encoding = self.tokenizer(section['text'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
            input_ids = encoding.input_ids[0]
            list_input_ids.append(input_ids)
            list_attention_masks.append(encoding.attention_mask[0])
            if input_ids[-1] == 0:
                input_ids[-1] = 1 
                # replace the last pad token by the end of sequence token. This is actually a bug in the tokenizer.
                # The tokenizer should not add a pad token at the end of the sequence. It should add the end of sequence token. (i.e., 1 = </s>)
            
            txt_splitted = section['text'].split("context: \\n")
            if len(txt_splitted) > 1 and txt_splitted[1] != '':
                # if there is context
                list_html_lines_ctx = txt_splitted[1].split('\\n')
            else:
                list_html_lines_ctx = []
            
            self.recursive_dom_tree_parsing(2, list_html_lines_ctx, [root_node], [section['title']], None, nodes, edges, sent_idx2node)
            
            
            # add hierarchical edges
            prev_root_node = stack[-1] if len(stack) > 0 else None
            if prev_root_node is not None:
                while self.gte_headers(root_node, prev_root_node) and len(stack) > 0:
                    stack.pop()
                    prev_root_node = stack[-1] if len(stack) > 0 else None

                if self.gt_headers(prev_root_node, root_node):
                    edge_type = prev_root_node + "_" + root_node
                    if edge_type not in edges:
                        edges[edge_type] = []
                    edges[edge_type].append((nodes[prev_root_node]-1, nodes[root_node]-1))
                    stack.append(root_node)
            else:
                stack.append(root_node)
            
            # add token nodes
            list_idx_nodes = [0]
            list_idx_nodes.extend((input_ids == self.id_slash_n).nonzero(as_tuple=True)[0].tolist())
            list_pads_occurences = (input_ids == self.id_pad).nonzero(as_tuple=True)[0].tolist()
            list_pads_and_sep = (input_ids == self.id_pad).nonzero(as_tuple=True)[0].tolist()
            list_pads_and_sep.extend((input_ids == self.id_sep).nonzero(as_tuple=True)[0].tolist())  
            if len(sent_idx2node) > 0: # if there is at least 1 sentence in the section
                i = 0
                if len(list_idx_nodes) > 1:
                    for i in range(len(list_idx_nodes)-1):
                        etype = sent_idx2node[i]['node_type'] + "_token"
                        if etype not in edges:
                            edges[etype] = []
                        for token_idx in range(list_idx_nodes[i],list_idx_nodes[i+1]):
                            t = token_idx + sec_idx*self.max_length # to add the offset
                            edges[etype].append((sent_idx2node[i]['node_idx'], t))
                            nodes['token'] += 1
                    # add token edges to the last node
                    i = i + 1
                    etype = sent_idx2node[i]['node_type'] + "_token"
                    if etype not in edges:
                        edges[etype] = []
                    for token_idx in range(list_idx_nodes[-1], list_pads_and_sep[0]):
                        t = token_idx + sec_idx*self.max_length # to add the offset
                        edges[etype].append((sent_idx2node[i]['node_idx'], t))
                        nodes['token'] += 1
                else:
                    # if there is no sentence in the section, only the title. eg: 'title: <h1> Leave </h1> context:'
                    i = 0
                    etype = sent_idx2node[i]['node_type'] + "_token"
                    if etype not in edges:
                        edges[etype] = []
                    for token_idx in range(list_idx_nodes[-1], list_pads_and_sep[0]):
                        t = token_idx + sec_idx*self.max_length # to add the offset
                        edges[etype].append((sent_idx2node[i]['node_idx'], t))
                        nodes['token'] += 1
            # add PAD tokens 
            # This will simplify the combination of section embeddings into the graph.
            # PAD nodes are connected to the root nodes (eg: h1)
            if len(list_pads_occurences) > 0:
                root_node = self.current_node(section['title'])
                etype = f"{root_node}_token"
                for token_idx in range(list_pads_occurences[0], len(input_ids)):
                    if etype not in edges:
                        edges[etype] = []
                    t = token_idx + sec_idx*self.max_length # to add the offset
                    # assert t == nodes['token']
                    edges[etype].append((nodes[root_node]-1, t))
                    nodes['token'] += 1     
            
        # add bidirectional edges
        reverse_edges = {}
        for etype in edges.keys():
            reverse_etype = "_".join(etype.split("_")[::-1])
            reverse_edges[reverse_etype] = [(v, u) for u, v in edges[etype]]
        edges.update(reverse_edges)
                
        # Create DGL graph
        graph_data = {}
        for edge_name, list_uv in edges.items():
            if len(list_uv) > 0:
                (src, dst) = edge_name.split('_')
                u, v = self.unzip_list(list_uv)
                graph_data[(src, edge_name, dst)] = (torch.tensor(u), torch.tensor(v))
        
        g = dgl.heterograph(graph_data)
        new_edges = self.add_sibling_edges(g)
        graph_data.update(new_edges)
        g = dgl.heterograph(graph_data)
        
        return (g, torch.stack(list_input_ids), torch.stack(list_attention_masks))
    
    def add_sibling_edges(self, g):
        graph_data = {}
        heading_nodes = ['h1', 'h2', 'h3', 'h4']
        for level in range(0, len(heading_nodes)-1):
            h = heading_nodes[level]
            if h in g.ntypes:
                # for each node, get its children. Those are siblings that need to be connected
                for h_node in g.nodes(h).tolist():
                    etype = (heading_nodes[level] + "_" + heading_nodes[level+1]) # h1_h2, h2_h3, h3_h4
                    if etype in g.etypes:
                        _, v = g.out_edges(h_node, etype=etype)
                        v = v.tolist()
                        if len(v) > 1:
                            sibling_nodes = []
                            for i in range(len(v)-1):
                                sibling_nodes.append((v[i], v[i+1]))
                            u, v = self.unzip_list(sibling_nodes)
                            child_lvl = heading_nodes[level+1]
                            child_etype = (child_lvl + "_" + child_lvl)
                            graph_data[(child_lvl, child_etype, child_lvl)] = (torch.tensor(u), torch.tensor(v))
        return graph_data
    
    def unzip_list(self, list_uv_pairs):
        list_u = []
        list_v = []
        for (u,v) in list_uv_pairs:
            list_u.append(u)
            list_v.append(v)
        return list_u, list_v

    def gte_headers(self, u, v):
        # greater than or equal to
        if u == v:
            return True
        elif u == "h1" and v != "h1":
            return True
        elif u == "h2" and v != "h1" and v != "h2":
            return True
        elif u == "h3" and v != "h1" and v != "h2" and v != "h3":
            return True
        elif u == "h4" and v != "h1" and v != "h2" and v != "h3" and v != "h4":
            return True
        else:
            return False
        
    def gt_headers(self, u, v):
        # greater than
        if u == v:
            return False
        elif u == "h1" and v != "h1":
            return True
        elif u == "h2" and v != "h1" and v != "h2":
            return True
        elif u == "h3" and v != "h1" and v != "h2" and v != "h3":
            return True
        elif u == "h4" and v != "h1" and v != "h2" and v != "h3" and v != "h4":
            return True
        else:
            return False
        
    def add_edge(self, edges, etype, u, v):
        if etype not in edges:
            edges[etype] = []
        edges[etype].append((u, v))
    
    # parse html DOM tree
    def recursive_dom_tree_parsing(self, sent_idx, section, list_parent_node, list_parent_lines, prev_line, nodes, edges, sent_idx2node):
        if len(section) == 0 or section == ['']:
            return       
        else:
            curr_node = self.current_node(section[0])
            sent_idx2node[sent_idx] = {"node_type": curr_node, "node_idx": nodes[curr_node]}
            if prev_line is None:
                # add edge to parent node
                parent_node = list_parent_node[-1]
                self.add_edge(edges, f"{parent_node}_{curr_node}", nodes[parent_node]-1, nodes[curr_node])
                nodes[curr_node] += 1
                return self.recursive_dom_tree_parsing(sent_idx+1, section[1:], list_parent_node, list_parent_lines, section[0], nodes, edges, sent_idx2node)
            elif prev_line is not None:
                prev_node = self.current_node(prev_line)
                if self.is_sibling_node(prev_node, curr_node):
                    # add edge to previous node
                    self.add_edge(edges, f"{prev_node}_{curr_node}", nodes[prev_node]-1, nodes[curr_node])
                    # add edge to parent node
                    parent_node = list_parent_node[-1]
                    self.add_edge(edges, f"{parent_node}_{curr_node}", nodes[parent_node]-1, nodes[curr_node])
                    # add node
                    nodes[curr_node] += 1
                    return self.recursive_dom_tree_parsing(sent_idx+1, section[1:], list_parent_node, list_parent_lines, section[0], nodes, edges, sent_idx2node)
                elif self.is_children_node(curr_node, prev_node):
                    # add edge to parent node
                    parent_node = prev_node
                    self.add_edge(edges, f"{parent_node}_{curr_node}", nodes[parent_node]-1, nodes[curr_node])
                    # add node
                    nodes[curr_node] += 1
                    list_parent_node.append(parent_node)
                    list_parent_lines.append(prev_line)
                    return self.recursive_dom_tree_parsing(sent_idx+1, section[1:], list_parent_node, list_parent_lines, section[0], nodes, edges, sent_idx2node)
                elif self.is_children_node(prev_node, curr_node):
                    if curr_node != list_parent_node[-1]:
                        # starts new section (eg: p -> h2)
                        return self.recursive_dom_tree_parsing(sent_idx, section, list_parent_node, list_parent_lines, None, nodes, edges, sent_idx2node)
                    else:
                        return self.recursive_dom_tree_parsing(sent_idx, section, list_parent_node[:-1], list_parent_lines[:-1], list_parent_lines[-1], nodes, edges, sent_idx2node)
                else:
                    return self.recursive_dom_tree_parsing(sent_idx, section, list_parent_node[:-1], list_parent_lines[:-1], list_parent_lines[-1], nodes, edges, sent_idx2node)
                    # add case for p li li tr tr (i.e., tr has to jump 2 levels)

    def add_html_tags2tokenizer(self, tokenizer):
        tokenizer.add_tokens(["<li>"])
        tokenizer.add_tokens(["</li>"])
        tokenizer.add_tokens(["<p>"])
        tokenizer.add_tokens(["</p>"])
        tokenizer.add_tokens(["<h1>"])
        tokenizer.add_tokens(["</h1>"])
        tokenizer.add_tokens(["<h2>"])
        tokenizer.add_tokens(["</h2>"])
        tokenizer.add_tokens(["<h3>"])
        tokenizer.add_tokens(["</h3>"])
        tokenizer.add_tokens(["<h4>"])
        tokenizer.add_tokens(["</h4>"])
        tokenizer.add_tokens(["<tr>"])
        tokenizer.add_tokens(["</tr>"])
        tokenizer.add_tokens(["\\n"])

    def current_node(self, line):
        if "<h1>" in line:
            return "h1"
        elif "<h2>" in line:
            return "h2"
        elif "<h3>" in line:
            return "h3"
        elif "<h4>" in line:
            return "h4"
        elif "<p>" in line:
            return "p"
        elif "<li>" in line:
            return "li"
        elif "<tr>" in line:
            return "tr"
        else:
            raise ValueError(f"Unknown node type {line}")
        
    def is_children_node(self, node, parent):
        if parent == "h1":
            return node in ["h2", "h3", "h4", "p", "li", "tr"]
        elif parent == "h2":
            return node in ["h3", "h4", "p", "li", "tr"]
        elif parent == "h3":
            return node in ["h4", "p", "li", "tr"]
        elif parent == "h4":
            return node in ["p", "li", "tr"]
        elif parent == "p":
            return node in ["li"]
        else:
            return False
        
    def is_sibling_node(self, prev_node, node):
        if prev_node in ["p", "tr"] and node in ["p", "tr"]:
            return True
        return prev_node == node

    def get_parent(self, node):
        if node == "h2":
            return "h1"
        elif node == "h3":
            return "h2"
        elif node == "h4":
            return "h3"
        elif node == "p":
            return "h4"
        elif node == "li":
            return "p" 
        elif node == "tr":
            return "p"       
        else:
            raise ValueError("No parent for this node")
        
# if __name__ == "__main__":
#     graph_builder = GraphBuilder()
#     with open('/home/puerto/projects/FiD/data/fid_format/train.json') as f:
#         train_data = json.load(f)
    
#     train_data = train_data[2170:2171]
#     train_data[0]['ctxs'] = train_data[0]['ctxs'][11:]
#     for sec in train_data[0]['ctxs']:
#         sec['text'] = "context: \\n ".join(sec['text'].split('context: '))
#     list_g = []
#     list_input_ids = []
#     list_attention_masks = []
#     for i in trange(len(train_data), desc="Creating graph", leave=True):
#         (g, input_ids, attention_mask) = graph_builder.create_graph(train_data[i]['ctxs'])
#         list_g.append(g)
#         list_input_ids.append(input_ids)
#         list_attention_masks.append(attention_mask)