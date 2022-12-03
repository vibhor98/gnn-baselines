"""Create graph dataset suitable for DGL library."""

import dgl
from dgl.data import DGLDataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
import os

class KialoDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='kialo')

    def process(self):
        dataset_path = '../serializedGraphs/'
        files = os.listdir(dataset_path)
        self.graphs = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model.max_seq_length = 256 #75
        print(self.model.max_seq_length)
        # self.model = CrossEncoder('../sbert_nli_model', num_labels=2, device="cpu")
        # self.model.model.base_model.eval()
        # self.model.model.base_model.to(self.model._target_device)

        for z, file in enumerate(files):
            print('Processing', z, file)

            data = pkl.load(open(dataset_path + file, 'rb'))
            num_nodes = len(data.node)-1
            # sentences = ['']*num_nodes
            sentences = [[] for _ in range(num_nodes)]
            node_labels = [-1]*num_nodes

            node_indices = [int(node_id.split('.')[1])-1 for node_id in data.node.keys()]
            node_indices.sort()
            node_indices.remove(-1)
            node_indx_map = {indx: i for i, indx in enumerate(node_indices)}

            edges_src = []
            edges_dst = []
            all_embeddings = []

            for node_id in data.node.keys():
                edge = data.edge[node_id]
                sent = data.node[node_id]['text']
                node_indx = int(node_id.split('.')[1]) - 1
                parent_indx = -1

                if node_indx != -1:
                    if len(edge.keys()) >= 1:
                        parent_node_id = list(edge.keys())[0]
                        parent_indx = int(parent_node_id.split('.')[1]) - 1

                    # sentences[node_indx_map[node_indx]].extend([sent1, sent2])
                    all_embeddings.extend(self.sbert_embeddings([sent]).to('cpu'))  #.detach().numpy())
                    # sentences[node_indx_map[node_indx]].append(sent2)

                    if parent_indx != -1:
                        if data.node[node_id]['relation'] == 1:
                            node_labels[node_indx_map[node_indx]] = 1
                            edges_src.append(node_indx_map[parent_indx])
                            edges_dst.append(node_indx_map[node_indx])
                        elif data.node[node_id]['relation'] == -1:
                            node_labels[node_indx_map[node_indx]] = 0
                            edges_src.append(node_indx_map[parent_indx])
                            edges_dst.append(node_indx_map[node_indx])
                    else:
                        # if data.node[node_id]['relation'] == 0:
                        node_labels[node_indx_map[node_indx]] = 1
            # Create a graph and add it to the list of graphs.
            g = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
            # node_features = self.model.encode(sentences)
            # node_features = self.sbert_embeddings(sentences)

            # node_features = np.array(all_embeddings)
            # g.ndata['feat'] = torch.from_numpy(node_features)
            g.ndata['feat'] = torch.stack(all_embeddings)
            g.ndata['label'] = torch.LongTensor(node_labels)
            self.graphs.append(g)
            # print(g.ndata['feat'].shape)
            # torch.Size([924, 384])

        print('Saving Kialo DGL Graphs...')
        dgl.save_graphs('kialo_dgl_graphs_sbert_joint.dgl', self.graphs)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)

    def sbert_embeddings(self, samples):
        # test_samples = [['This is for test.', 'No, this is for train.']]
        # test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=1)
            # collate_fn=self.model.smart_batching_collate_text_only)
        # print(dir(model))
        # print(model.config)
        # print(model.model.base_model)
        # print(model.model.classifier)
        all_embeddings = []
        # for features in test_dataloader:
        # hidden_features = self.model.model.base_model(**features, return_dict=True) # , output_hidden_states=True)
        hidden_features = self.model.encode(samples, convert_to_tensor=True)
        # embeddings = hidden_features.last_hidden_state[:, 0, :]
        # embeddings = self.mean_pooling(hidden_features.last_hidden_state, features['attention_mask'])
        embeddings = F.normalize(hidden_features, p=2, dim=1)
        # print(embed.size())
        # print(embed)
        return embeddings
        #     all_embeddings.extend(embeddings)
        # return np.array(all_embeddings)

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

dataset = KialoDataset()
#graph = dataset[0]

#print(graph)
