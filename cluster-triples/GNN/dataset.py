import os

import torch
from torch_geometric.data import Data, Dataset
from torch.nn import functional

class GraphData(Dataset):
    def __init__(self, root: str):
        self.node_features = []

        for i in range(71):
            self.node_features.append(functional.one_hot(torch.tensor(i), num_classes=71))
        
        self.node_features = torch.stack(self.node_features).to(torch.float)

        super().__init__(root)

    
    def get_test(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data, self.processed_file_names[idx]

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data

    def len(self):
        return len(self.processed_file_names)  #changed this line

    @property
    def raw_file_names(self):
        raw_file_names = list(os.listdir(self.raw_dir))
        return raw_file_names
    
    @property
    def processed_file_names(self):
        
        processed_file_names = list(os.listdir(self.processed_dir))
        return processed_file_names


    def process(self):
        
        for file in self.raw_file_names:
            self._process_one_step(file)

    def _process_one_step(self, file_path):

        out_path = self.processed_dir + "/"+ file_path + ".pt"

        with open(self.raw_dir + "/" +file_path, 'r') as fp:

            multi_edge_detector = set()  # CURBS ALL MULTIPLE EDGES IN A GRAPH

            edge_src = []
            edge_dst = []

            for edges in fp.readlines():
                if edges in multi_edge_detector:
                    continue
                
                multi_edge_detector.add(edges)

                e_1 = int(edges.split(" ")[0])
                e_2 = int(edges.split(" ")[1].split('\n')[0])
 
                edge_src.append(torch.tensor(e_1))
                edge_dst.append(torch.tensor(e_2))
 
        
        edge_src = torch.stack(edge_src)
        edge_dst = torch.stack(edge_dst)
        
        edge_index = torch.stack([edge_src, edge_dst]).to(torch.long)
        pos_edge_index = edge_index[:,0:4]

        data_object = Data(
            x=self.node_features,
            edge_index=edge_index,
            pos_edge_index=pos_edge_index
        )

        torch.save(data_object, out_path)
    

