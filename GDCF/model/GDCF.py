import logging
import math
import numpy as np
import torch
import torch.nn as nn
from modules.memory import ExpMemory_lambs, Memory_lambs
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module


class Encoder(nn.Module):
    def __init__(self, n_nodes, node_features, embedding_dimension, memory_dimension, message_dimension, lambs, device,
                 output, init_lamb=0.5):
        super(Encoder, self).__init__()
        self.n_nodes = n_nodes
        self.node_features = node_features
        self.embedding_dimension = embedding_dimension
        self.output = output
        self.memory_dimension = memory_dimension
        self.message_dimension = message_dimension
        self.device = device
        self.lambs = torch.Tensor(lambs).to(self.device) * self.output
        self.lamb_len = self.lambs.shape[0]
        raw_message_dimension = self.memory_dimension * self.lamb_len + self.node_features.shape[1]
        self.memory = ExpMemory_lambs(n_nodes=self.n_nodes,
                                      memory_dimension=self.memory_dimension,
                                      lambs=self.lambs,
                                      device=self.device)  # (3, nodes, raw_message_dim)
        self.message_aggregator = get_message_aggregator(aggregator_type="exp_lambs", device=self.device,
                                                         embedding_dimension=memory_dimension)
        self.message_function = get_message_function(module_type="mlp",
                                                     raw_message_dimension=raw_message_dimension,
                                                     message_dimension=self.message_dimension)
        self.memory_updater = get_memory_updater(module_type="exp_lambs",
                                                 memory=self.memory,
                                                 message_dimension=self.message_dimension,
                                                 memory_dimension=self.lambs,
                                                 device=self.device)
        self.exp_embedding = get_embedding_module(module_type="exp_lambs",
                                                  n_node_features=-1)
        self.iden_embedding = get_embedding_module(module_type="identity",
                                                   n_node_features=self.memory_dimension)
        self.static_embedding = nn.Embedding(self.n_nodes, self.embedding_dimension)

        self.n_virtuals = math.ceil(math.sqrt(self.n_nodes))
        self.virtual_memory = Memory_lambs(n_nodes=self.n_virtuals,
                                          memory_dimension=self.memory_dimension,
                                          lambs=self.lambs,
                                          device=self.device)

        self.virtual_memory_updater = get_memory_updater("exp_lambs", self.virtual_memory, self.message_dimension,
                                                        self.lambs,
                                                        self.device)

        self.message_multi_head = raw_message_dimension

        self.A_r = nn.Parameter(
            (torch.ones([self.n_nodes, self.n_virtuals]) + torch.rand([self.n_nodes, self.n_virtuals]) / 10).to(
                self.device), requires_grad=True).to(self.device)
        self.ff_r = torch.nn.Sequential(
            torch.nn.Linear(self.message_multi_head, self.message_multi_head, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.message_multi_head, self.message_dimension, bias=True),
            torch.nn.LeakyReLU()
        )
        self.embedding_transform = torch.nn.Sequential(
            torch.nn.Linear(memory_dimension * self.lamb_len, memory_dimension, bias=True),
            torch.nn.LeakyReLU()
        )
        self.spatial_transform = torch.nn.Sequential(
            torch.nn.Linear(memory_dimension * 2, embedding_dimension, bias=True),
            torch.nn.LeakyReLU()
        )
        self.lamb = nn.Parameter(torch.Tensor([init_lamb]), requires_grad=True)

    def forward(self, source_nodes, target_nodes, timestamps_batch_torch, now_time, predict_IND):
        # 1. Get node messages from last updated memories
        memory = self.memory.get_memory()
        target_embeddings = self.exp_embedding.compute_embedding(memory=memory,
                                                                 nodes=target_nodes)  # (nodes, l * memory)
        # Compute node_level messages
        raw_messages = self.get_raw_messages(source_nodes,
                                             target_embeddings,
                                             self.node_features[target_nodes],
                                             timestamps_batch_torch)  # (nodes, l * memory + feature)
        unique_nodes, unique_raw_messages, unique_timestamps = self.message_aggregator.aggregate(source_nodes,
                                                                                                 raw_messages,
                                                                                                 self.lambs)  # unique_raw_messages: (nodes, l, raw_message_dim)
        unique_messages = torch.cat(
            [self.message_function.compute_message(unique_raw_messages[:, :, :-1]), unique_raw_messages[:, :, -1:]],
            dim=-1)  # (nodes, l, message_dim)

        # 2. Compute messages for different levels
        A = torch.softmax(self.A_r, dim=0)[unique_nodes]
        B = (unique_raw_messages[:, :, :-1] / unique_raw_messages[:, :, -1:]).reshape(len(unique_nodes), self.lamb_len, -1)
        print("A", A.shape)
        print("B", B.shape)
        virtual_messages_mid = torch.einsum("nr,nlf->rlf", A,B)

        virtual_messages = self.ff_r(virtual_messages_mid)  # (n_virtuals, lambs, message_dim)

        # 3. Update memories
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)
        recent_node_embeddings = self.exp_embedding.compute_embedding(memory=updated_memory,
                                                                      nodes=list(range(self.n_nodes)))
        updated_virtual_memory, _ = self.virtual_memory_updater.get_updated_memory(list(range(self.n_virtuals)),
                                                                                 virtual_messages,
                                                                                 timestamps=now_time)
        recent_virtual_embeddings = self.iden_embedding.compute_embedding(memory=updated_virtual_memory,
                                                                         nodes=list(range(self.n_virtuals))).reshape(
            [self.n_virtuals, -1])

        r2n = torch.softmax(self.A_r, dim=1)  # n * r

        virtual_node_embeddings = torch.mm(r2n, recent_virtual_embeddings)
        dynamic_embeddings = torch.cat([recent_node_embeddings, virtual_node_embeddings], dim=0)
        embeddings = self.lamb * self.static_embedding.weight + (1 - self.lamb) * self.spatial_transform(
            self.embedding_transform(
                dynamic_embeddings).reshape([2, self.n_nodes, self.memory_dimension]).permute([1, 0, 2]).reshape(
                [self.n_nodes, -1]))

        self.memory_updater.update_memory(unique_nodes, unique_messages, timestamps=unique_timestamps)
        self.virtual_memory_updater.update_memory(list(range(self.n_virtuals)), virtual_messages, timestamps=now_time)
        #embeddings shape [63, 128] [nodes , embedding_dim]
        return embeddings

    def get_raw_messages(self, source_nodes, target_embeddings, node_features, edge_times):
        source_message = torch.cat(
            [target_embeddings, node_features, torch.ones([target_embeddings.shape[0], 1]).to(self.device)], dim=1)
        messages = dict()
        unique_nodes = np.unique(source_nodes)
        for node_i in unique_nodes:
            ind = np.arange(source_message.shape[0])[source_nodes == node_i]
            messages[node_i] = [source_message[ind], edge_times[ind]]
        return messages

    def init_memory(self):
        self.memory.__init_memory__()
        self.virtual_memory.__init_memory__()

    def backup_memory(self):
        return [self.memory.backup_memory(), self.virtual_memory.backup_memory()]

    def restore_memory(self, memory):
        self.memory.restore_memory(memory[0])
        self.virtual_memory.restore_memory(memory[1])

    def detach_memory(self):
        self.memory.detach_memory()
        self.virtual_memory.detach_memory()


class Decoder(nn.Module):
    def __init__(self, embedding_dim, n_nodes, device):
        super(Decoder, self).__init__()
        self.device = device
        self.n_nodes = n_nodes
        self.linear1 = nn.Linear(in_features=1, out_features=1)
        self.linear2 = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, int(embedding_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2), 1)
        )

    def forward(self, embeddings, o_nodes, d_nodes, time_diffs, edge_ind, od_matrix, output_len):
        pos_value = edge_ind
        pos_pairs = torch.cat([embeddings[o_nodes], embeddings[d_nodes]], dim=-1) * self.linear1(
            time_diffs.unsqueeze(-1) / output_len)
        od_matrix[od_matrix == 0] = 1

        neg_mask = np.ones([self.n_nodes, self.n_nodes])
        neg_mask[o_nodes, d_nodes] = 0
        neg_mask = neg_mask.astype(bool)
        neg_o = np.arange(self.n_nodes).repeat([self.n_nodes])[neg_mask.reshape(-1)]
        neg_d = np.arange(self.n_nodes).repeat(self.n_nodes).reshape([self.n_nodes, self.n_nodes]).transpose(1,
                                                                                                             0).reshape(
            -1)[neg_mask.reshape(-1)]
        neg_pairs = torch.cat([embeddings[neg_o], embeddings[neg_d]],
                              dim=-1)

        normalizer = torch.Tensor(od_matrix[np.concatenate([o_nodes, neg_o]), np.concatenate([d_nodes, neg_d])]).to(
            self.device)

        out = self.linear2(torch.cat([pos_pairs, neg_pairs], dim=0)).reshape(-1)
        truth = torch.cat([pos_value, torch.zeros(np.sum(neg_mask)).to(self.device)])
        nll = torch.sum(torch.pow(out - truth, 2) / normalizer) / (self.n_nodes * self.n_nodes)
        return nll


class GDCF(nn.Module):
    def __init__(self, device,
                 n_nodes=268, node_features=None,
                 message_dimension=64, memory_dimension=64, lambs=None,
                 output=30):
        super(GDCF, self).__init__()
        if lambs is None:
            lambs = [1]
        self.logger = logging.getLogger(__name__)
        self.device = device
        node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        embedding_dimension = memory_dimension * 1
        self.encoder = Encoder(n_nodes, node_raw_features, embedding_dimension, memory_dimension,
                               message_dimension, lambs, device,
                               output, init_lamb=0.1)
        self.output = output
        self.reconstruct = Decoder(embedding_dimension, n_nodes, device)

    #这个encoder阶段才会用到
    def compute_IND(self, o_nodes, d_nodes, timestamps_batch_torch, now_time, time_diffs, edge_index, od_matrix,
                    predict_IND=True):
        embeddings = self.encoder(o_nodes, d_nodes, timestamps_batch_torch, now_time, predict_IND)
        nll = None
        if predict_IND:
            nll = self.reconstruct(embeddings, o_nodes, d_nodes, time_diffs,
                                                                edge_index, od_matrix, self.output)

        return nll, embeddings #loss 和 更新后的node embedding embedding是每一个batch都要更新的，loss是 now_time % day_cycle < day_end时候才会去计算。

    def init_memory(self):
        self.encoder.init_memory()

    def backup_memory(self):
        return self.encoder.backup_memory()

    def restore_memory(self, memories):
        self.encoder.restore_memory(memories)

    def detach_memory(self):
        self.encoder.detach_memory()
