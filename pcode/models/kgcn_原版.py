import random
from collections import defaultdict

import torch
import torch.nn.functional as F

__all__ = ["kgcn", "kgcn_aggregate", "kgcn_kg"]

from torch import nn

from pcode.utils.auto_distributed import recv_list, send_list


class Aggregator(torch.nn.Module):
    '''
    Aggregator class根据指定的聚合策略计算当前节点与邻居节点特征的融合。
    Mode in ['sum', 'concat', 'neighbor']
    '''

    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim  # 16
        # 根据 aggregator 的选择，初始化了不同的 Linear 层
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True) #当前节点和邻居节点特征拼接
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator = aggregator

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, act):
        # 前向传播函数,act：激活函数（如 ReLU、Sigmoid 等）
        # self_vectors：当前节点的特征向量（形状为 [batch_size, 1, dim]）。
        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size

        # 对邻居特征进行加权聚合
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).reshape((-1, self.dim)) #直接相加

        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1) # 拼接
            output = output.reshape((-1, 2 * self.dim))

        else:
            output = neighbors_agg.reshape((-1, self.dim)) # 只使用聚合后的邻居特征

        output = self.weights(output) #使用 self.weights（Linear 层）对聚合后的特征进行线性变换，调整特征的维度
        # 使用提供的激活函数 act 对输出进行激活处理
        return act(output.reshape((self.batch_size, -1, self.dim)))

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        '''
        This aims to aggregate neighbor vectors邻居特征聚合函数，根据用户嵌入与邻居之间的关系，计算加权聚合
        '''
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]重塑为形状使其能够与邻居关系矩阵进行逐元素乘法。
        user_embeddings = user_embeddings.reshape((self.batch_size, 1, 1, self.dim))

        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        # 计算用户与邻居之间的关系得分
        # print('user_embeddings',user_embeddings.shape)
        # print('neighbor_relations',neighbor_relations.shape)
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim=-1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1) #归一化

        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim=-1)

        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim=2)

        return neighbors_aggregated


class KGCN(torch.nn.Module):
    def __init__(self, num_usr, num_ent, num_rel, kg, args, device):
        super(KGCN, self).__init__()
        self.num_usr = num_usr
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size  # 采样邻居个数
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)  # 相加然后过线性层
        self.init = torch.nn.init.xavier_normal_
        self._gen_adj()  # 对KG中的每一个head,固定采样n_neighbor个邻居节点和关系

        self.usr = torch.zeros(num_usr, args.dim).to(self.device)
        self.ent = torch.zeros(num_ent, args.dim).to(self.device)
        self.rel = torch.zeros(num_rel, args.dim).to(self.device)
        self.init(self.usr)
        self.init(self.ent)
        self.init(self.rel)

        self._init_trained_tracker()

    def _init_trained_tracker(self):
        self.trained = defaultdict(list)
        self.trained['user_embeddings_grad'] = 0
        self.trained['entities_embeddings_grad'] = 0
        self.trained['relations_embeddings_grad'] = 0

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)

        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)

            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])

    def forward(self, data_batch, entities=None, relations=None):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        (u, v) = data_batch
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        u = u.reshape((-1, 1))
        v = v.reshape((-1, 1))

        self.usr.requires_grad = True
        self.ent.requires_grad = True
        self.rel.requires_grad = True

        # [batch_size, dim]
        user_embeddings = self.usr[u].squeeze(dim=1)
        if entities == None and relations == None:
            entities, relations = self._get_neighbors(v)  # 对每一个user-item的item,取item的n_iter层邻居

        item_embeddings = self._aggregate(user_embeddings, entities, relations)  # 单层加权求和

        scores = (user_embeddings * item_embeddings).sum(dim=1)
        return torch.sigmoid(scores)

    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).reshape((self.batch_size, -1)).to(
                self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).reshape((self.batch_size, -1)).to(
                self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent[entity] for entity in entities]
        relation_vectors = [self.rel[relation] for relation in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].reshape((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].reshape((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].reshape((self.batch_size, self.dim))

    def request_neighbors(self, user_id, item_id):
        with torch.no_grad():
            # user_embeddings
            [user_embeddings] = recv_list(0)
            self.user_id = user_id
            self.usr[user_id] = user_embeddings

            # item_embeddings
            self.neighbor_entities, self.neighbor_relations = self.get_neighbor_id(item_id)
            send_list([self.neighbor_entities, self.neighbor_relations], 0)
            entities_embeddings, relations_embeddings = recv_list(0)
            self.ent[self.neighbor_entities] = entities_embeddings
            self.rel[self.neighbor_relations] = relations_embeddings

    def get_neighbor_id(self, item_id):
        neighbor_entities = torch.flatten(torch.LongTensor(self.adj_ent[item_id])).unique()
        neighbor_relations = torch.flatten(torch.LongTensor(self.adj_rel[item_id])).unique()
        return neighbor_entities, neighbor_relations

    def distribute_neighbors(self, selected_client_ids):
        with torch.no_grad():
            self.trained['user_id'] += [torch.LongTensor([id - 1]) for id in selected_client_ids]
            # user_embeddings
            user_embeddings = [self.usr[torch.tensor(id - 1).to(self.device)] for id in selected_client_ids]
            for worker, embeddings in enumerate(user_embeddings, 1):
                send_list([user_embeddings[worker - 1]], worker)

            # item_embeddings

            for worker, embeddings in enumerate(user_embeddings, 1):
                [entities, relations] = recv_list(worker)
                self.trained['entities_id'].append(entities)
                self.trained['relations_id'].append(relations)
                send_list([self.ent[entities], self.rel[relations]], worker)

    def upload_selected(self):
        with torch.no_grad():
            user_embeddings_grad = self.usr.grad[self.user_id]
            entities_embeddings_grad = self.ent.grad[self.neighbor_entities]
            relations_embeddings_grad = self.rel.grad[self.neighbor_relations]
            send_list([user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad], 0)

    def upload(self):
        with torch.no_grad():
            user_embeddings_grad = self.usr.grad
            entities_embeddings_grad = self.ent.grad
            relations_embeddings_grad = self.rel.grad
            send_list([user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad], 0)

    def receive_selected(self, selected_client_ids):
        with torch.no_grad():
            for worker, selected_client_id in enumerate(selected_client_ids, 1):
                [user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad] = recv_list(worker,
                                                                                                        store_device=torch.device(
                                                                                                            'cpu'))
                self.trained['user_embeddings_grad'].append(user_embeddings_grad)
                self.trained['entities_embeddings_grad'].append(entities_embeddings_grad)
                self.trained['relations_embeddings_grad'].append(relations_embeddings_grad)

    def receive(self, selected_client_ids):
        with torch.no_grad():
            for worker, selected_client_id in enumerate(selected_client_ids, 1):
                [user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad] = recv_list(worker,
                                                                                                        store_device=torch.device(
                                                                                                            'cpu'))
                self.trained['user_embeddings_grad'] += user_embeddings_grad
                self.trained['entities_embeddings_grad'] += entities_embeddings_grad
                self.trained['relations_embeddings_grad'] += relations_embeddings_grad

    def update_selected(self):
        with torch.no_grad():
            device = torch.device('cpu')
            self.trained['user_id'] = torch.concat(self.trained['user_id']).to(device)
            self.trained['entities_id'] = torch.concat(self.trained['entities_id']).to(device)
            self.trained['relations_id'] = torch.concat(self.trained['relations_id']).to(device)
            self.trained['user_embeddings_grad'] = torch.vstack(self.trained['user_embeddings_grad']).to(device)
            self.trained['entities_embeddings_grad'] = torch.vstack(self.trained['entities_embeddings_grad']).to(device)
            self.trained['relations_embeddings_grad'] = torch.vstack(self.trained['relations_embeddings_grad']).to(
                device)
            self.trained_usr = torch.zeros_like(self.usr, device='cpu')
            self.trained_ent = torch.zeros_like(self.ent, device='cpu')
            self.trained_rel = torch.zeros_like(self.rel, device='cpu')

            self.trained_usr.scatter_reduce_(0, self.trained['user_id'].expand(self.trained_usr.shape[1],
                                                                               len(self.trained['user_id'])).T,
                                             self.trained['user_embeddings_grad'], "sum", include_self=False)
            self.trained_ent.scatter_reduce_(0, self.trained['entities_id'].expand(self.trained_ent.shape[1],
                                                                                   len(self.trained['entities_id'])).T,
                                             self.trained['entities_embeddings_grad'], "sum", include_self=False)
            self.trained_rel.scatter_reduce_(0, self.trained['relations_id'].expand(self.trained_rel.shape[1],
                                                                                    len(self.trained[
                                                                                            'relations_id'])).T,
                                             self.trained['relations_embeddings_grad'], "sum", include_self=False)
            batch_size = len(self.trained['user_id'])
            self.trained_usr = self.trained_usr.to(self.device) / batch_size
            self.trained_ent = self.trained_ent.to(self.device) / batch_size
            self.trained_rel = self.trained_rel.to(self.device) / batch_size
            self.usr.grad = torch.zeros_like(self.trained_usr)
            self.ent.grad = torch.zeros_like(self.trained_ent)
            self.rel.grad = torch.zeros_like(self.trained_rel)
            self.usr.grad = torch.where(self.trained_usr == self.usr.grad, self.usr.grad, self.trained_usr)
            self.ent.grad = torch.where(self.trained_ent == self.ent.grad, self.ent.grad, self.trained_ent)
            self.rel.grad = torch.where(self.trained_rel == self.rel.grad, self.rel.grad, self.trained_rel)

            self._init_trained_tracker()

    def update(self):
        with torch.no_grad():
            device = torch.device('cpu')
            self.trained['user_id'] = torch.concat(self.trained['user_id']).to(device)
            self.trained['entities_id'] = torch.concat(self.trained['entities_id']).to(device)
            self.trained['relations_id'] = torch.concat(self.trained['relations_id']).to(device)
            batch_size = len(self.trained['user_id'])
            self.usr.grad = self.trained['user_embeddings_grad'].to(self.device) / batch_size
            self.ent.grad = self.trained['entities_embeddings_grad'].to(self.device) / batch_size
            self.rel.grad = self.trained['relations_embeddings_grad'].to(self.device) / batch_size
            self._init_trained_tracker()


class KGCN_E(torch.nn.Module):
    def __init__(self, num_usr, num_ent, num_rel, kg, args, device):
        super(KGCN_E, self).__init__()
        self.num_usr = num_usr
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size  # 采样邻居个数
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)  # 相加然后过线性层
        self._gen_adj()  # 对KG中的每一个head,固定采样n_neighbor个邻居节点和关系
        self.init = torch.nn.init.xavier_normal_

        self.usr = nn.Embedding(num_usr, args.dim)
        self.ent = nn.Embedding(num_ent, args.dim)
        self.rel = nn.Embedding(num_rel, args.dim)
        self.init(self.usr.weight)
        self.init(self.ent.weight)
        self.init(self.rel.weight)

        self.trained = defaultdict(list)

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)

        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)

            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])

    def forward(self, data_batch, entities=None, relations=None):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        (u, v) = data_batch
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        u = u.reshape((-1, 1))
        v = v.reshape((-1, 1))

        self.usr.requires_grad = True
        self.ent.requires_grad = True
        self.rel.requires_grad = True

        # [batch_size, dim]
        user_embeddings = self.usr(u).squeeze(dim=1)
        if entities == None and relations == None:
            entities, relations = self._get_neighbors(v)  # 对每一个user-item的item,取item的n_iter层邻居

        item_embeddings = self._aggregate(user_embeddings, entities, relations)  # 单层加权求和

        scores = (user_embeddings * item_embeddings).sum(dim=1)

        return torch.sigmoid(scores)

    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).reshape((self.batch_size, -1)).to(
                self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).reshape((self.batch_size, -1)).to(
                self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].reshape((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].reshape((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].reshape((self.batch_size, self.dim))

    def request_neighbors(self, user_id, item_id):
        with torch.no_grad():
            # user_embeddings
            [user_embeddings] = recv_list(0)
            self.user_id = user_id
            self.usr.weight[user_id] = user_embeddings

            # item_embeddings
            self.neighbor_entities, self.neighbor_relations = self.get_neighbor_id(item_id)
            send_list([self.neighbor_entities, self.neighbor_relations], 0)
            entities_embeddings, relations_embeddings = recv_list(0)
            self.ent.weight[self.neighbor_entities] = entities_embeddings
            self.rel.weight[self.neighbor_relations] = relations_embeddings

    def get_neighbor_id(self, item_id):
        neighbor_entities = torch.flatten(torch.LongTensor(self.adj_ent[item_id]))
        neighbor_entities = torch.concat([neighbor_entities, item_id]).unique()
        neighbor_relations = torch.flatten(torch.LongTensor(self.adj_rel[item_id])).unique()
        return neighbor_entities, neighbor_relations

    def distribute_neighbors(self, selected_client_ids):
        with torch.no_grad():
            self.trained['user_id'] += [torch.LongTensor([id - 1]) for id in selected_client_ids]
            # user_embeddings
            user_embeddings = [self.usr(torch.tensor(id - 1)) for id in selected_client_ids]
            for worker, embeddings in enumerate(user_embeddings, 1):
                send_list([user_embeddings[worker - 1]], worker)

            # item_embeddings

            for worker, embeddings in enumerate(user_embeddings, 1):
                [entities, relations] = recv_list(worker)
                self.trained['entities_id'].append(entities)
                self.trained['relations_id'].append(relations)
                send_list([self.ent(entities), self.rel(relations)], worker)

    def upload(self):
        with torch.no_grad():
            user_embeddings_grad = self.usr.weight.grad[self.user_id]
            entities_embeddings_grad = self.ent.weight.grad[self.neighbor_entities]
            relations_embeddings_grad = self.rel.weight.grad[self.neighbor_relations]
            send_list([user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad], 0)

    def receive(self, selected_client_ids):
        with torch.no_grad():
            for worker, selected_client_id in enumerate(selected_client_ids, 1):
                [user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad] = recv_list(worker,
                                                                                                        store_device=torch.device(
                                                                                                            'cpu'))
                self.trained['user_embeddings_grad'].append(user_embeddings_grad)
                self.trained['entities_embeddings_grad'].append(entities_embeddings_grad)
                self.trained['relations_embeddings_grad'].append(relations_embeddings_grad)

    def update(self):
        with torch.no_grad():
            device = torch.device('cpu')
            self.trained['user_id'] = torch.concat(self.trained['user_id']).to(device)
            self.trained['entities_id'] = torch.concat(self.trained['entities_id']).to(device)
            self.trained['relations_id'] = torch.concat(self.trained['relations_id']).to(device)
            self.trained['user_embeddings_grad'] = torch.vstack(self.trained['user_embeddings_grad']).to(device)
            self.trained['entities_embeddings_grad'] = torch.vstack(self.trained['entities_embeddings_grad']).to(device)
            self.trained['relations_embeddings_grad'] = torch.vstack(self.trained['relations_embeddings_grad']).to(
                device)
            self.trained_usr = torch.zeros_like(self.usr.weight, device='cpu')
            self.trained_ent = torch.zeros_like(self.ent.weight, device='cpu')
            self.trained_rel = torch.zeros_like(self.rel.weight, device='cpu')

            self.trained_usr.scatter_reduce_(0, self.trained['user_id'].expand(self.trained_usr.shape[1],
                                                                               len(self.trained['user_id'])).T,
                                             self.trained['user_embeddings_grad'], "sum", include_self=False)
            self.trained_ent.scatter_reduce_(0, self.trained['entities_id'].expand(self.trained_ent.shape[1],
                                                                                   len(self.trained['entities_id'])).T,
                                             self.trained['entities_embeddings_grad'], "sum", include_self=False)
            self.trained_rel.scatter_reduce_(0, self.trained['relations_id'].expand(self.trained_rel.shape[1],
                                                                                    len(self.trained[
                                                                                            'relations_id'])).T,
                                             self.trained['relations_embeddings_grad'], "sum", include_self=False)
            batch_size = len(self.trained['user_id'])
            self.trained_usr = (self.trained_usr / batch_size).to(device)
            self.trained_ent = (self.trained_ent / batch_size).to(device)
            self.trained_rel = (self.trained_rel / batch_size).to(device)
            self.usr.weight.grad = torch.zeros_like(self.trained_usr)
            self.ent.weight.grad = torch.zeros_like(self.trained_ent)
            self.rel.weight.grad = torch.zeros_like(self.trained_rel)
            self.usr.weight.grad = torch.where(self.trained_usr == self.usr.weight.grad, self.usr.weight.grad,
                                               self.trained_usr)
            self.ent.weight.grad = torch.where(self.trained_ent == self.ent.weight.grad, self.ent.weight.grad,
                                               self.trained_ent)
            self.rel.weight.grad = torch.where(self.trained_rel == self.rel.weight.grad, self.rel.weight.grad,
                                               self.trained_rel)

            self.trained = defaultdict(list)

    def parameters(self, embeddings=False):
        for name, param in self.named_parameters():
            if embeddings or 'aggregator' in name:
                yield param


class KGCN_kg(torch.nn.Module):
    # 基于知识图谱的图神经网络
    def __init__(self, num_usr, num_ent, num_rel, kg, args, device):
        super(KGCN_kg, self).__init__()
        self.num_usr = num_usr
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter #迭代次数
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size  # 采样邻居个数
        self.kg = kg
        self.device = device
        # 用于聚合邻居节点特征的模块
        self.aggregator = KGCN_aggregator(args.batch_size, args.dim, args.neighbor_sample_size, args.aggregator,
                                          args.n_iter)
        self._gen_adj()  # 对KG中的每一个head,固定采样n_neighbor个邻居节点和关系
        self.init = torch.nn.init.xavier_normal_ #使用 Xavier 初始化方法来初始化嵌入层的权重
        self.neighbor_embeddings_dict = {} #字典：用于存储每个节点的邻居嵌入
        self.id_map = {}

        self.usr = nn.Embedding(num_usr, args.dim) #定义了一个用户嵌入层，num_usr 是用户的数量，args.dim 是嵌入的维度
        self.ent = nn.Embedding(num_ent, args.dim) #定义了一个实体嵌入层，num_ent 是实体的数量
        self.rel = nn.Embedding(num_rel, args.dim) #定义了一个关系嵌入层，num_rel 是关系的数量
        self.init(self.usr.weight) #初始化上述嵌入层的权重
        self.init(self.ent.weight)
        self.init(self.rel.weight)

        # 定义了一个默认字典 trained，用于存储训练过程中的一些中间数据。defaultdict 是 Python 的一个特殊字典类型，当访问一个不存在的键时，它会返回一个默认值（这里是空列表 list）
        self.trained = defaultdict(list)

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples为每个实体（e）选取一定数量的邻居，并记录这些邻居的实体和关系信息
        '''
        # 注册一个不会进行梯度更新的张量
        self.register_buffer('adj_ent', torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)) #[num_ent, n_neighbor]，用于存储每个实体的邻居实体的 ID。
        self.register_buffer('adj_rel', torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)) #用于存储每个实体的邻居关系的 ID。

        for e in self.kg:
            # self.kg[e]：表示知识图谱中实体 e 的所有邻居信息
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor) #随机抽取 n_neighbor 个邻居
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor) #使用 random.choices 来从原始邻居列表中重复采样

            #  neighbors 是一个列表，其中每个元素是一个元组 (relID, entID)
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors]) #实体 e 的n_neighbor个邻居实体的 ID
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors]) #实体 e与每个邻居实体相对应的关系 ID

    def _get_embeddings(self, user_id, dataset,batch_size, request_by_client=False):
        '''用于在每次联邦学习中获取用户嵌入向量以及与用户相关的邻居实体和关系的嵌入。这些嵌入最终会被分发给客户端以支持进一步的个性化训练或预测'''
        with torch.no_grad():
            if request_by_client:
                pass
            else:
                # 检查是否已有缓存
                # print(dataset.df[dataset.df['userID'] == user_id])
                # print(368 in dataset.df["userID"].values) 
                if not hasattr(self, "dataset_dict"):
                    self.dataset_dict = {
                        idx[0]: [torch.tensor(df["itemID"].values), torch.tensor(df["label"].values, dtype=torch.float)]
                        for idx, df in dataset.df.groupby(["userID"])}
                    # print(self.dataset_dict.keys())  # 打印 dataset_dict 的所有 user_id 键
                item_ids = self.dataset_dict[user_id][0]
                target = self.dataset_dict[user_id][1]
            if batch_size == None or batch_size>item_ids.size(0):
                self.batch_size = item_ids.size(0)
            else:
                idx = torch.randperm(item_ids.size(0))
                item_ids = item_ids[idx[:batch_size]]
                target = target[idx[:batch_size]]
                self.batch_size = batch_size
            # change to [batch_size, 1]
            item_ids = item_ids.clone().reshape((-1, 1))
            # [batch_size, dim]

            # user_id通过嵌入层，得到用户嵌入矩阵[1,dim]
            user_embeddings = self.usr(torch.tensor([user_id]))
            # print('item_ids-----------',item_ids)
            entities, relations = self._get_neighbors(item_ids)  # 对每一个user-item的item,取item的n_iter层邻居
            # print('entities1-----------',entities)
            entity_set = torch.concat([entity.unique() for entity in entities]).unique()
            entity_map = {int(entity): i for i, entity in enumerate(entity_set)}
            # for entity in entities:
            #     print(entity,entity.shape)
            
            entities = [entity.map_(entity, lambda x, y: entity_map[x]) for entity in entities] 
            # for entity in entities:
            #     print(entity,entity.shape)
            # print('entity_set-----------',entity_set)
            # print('entities2-----------',entities)
#             形状为 (N, d) 的张量，其中 N 是实体的数量，d 是嵌入向量的维度。
            entity_embeddings = torch.vstack([self.ent(entity) for entity in entity_set])
            # entity_embeddings = torch.vstack([self.ent(entity) for entity in entities])

            relation_set = torch.concat([relation.unique() for relation in relations]).unique()
            relation_map = {int(relation): i for i, relation in enumerate(relation_set)}
            relations = [relation.map_(relation, lambda x, y: relation_map[x]) for relation in relations]
            relation_embeddings = torch.vstack([self.rel(relation) for relation in relation_set])
            # relation_embeddings = torch.vstack([self.rel(relation) for relation in relations])

            self.id_map[user_id] = [list(entity_map.keys()), list(relation_map.keys())]
            # print('entity_embeddings',entity_embeddings)
            return [user_embeddings, entities, entity_embeddings,
                    relations, relation_embeddings, target]

    def _all_embeddings(self, user_id, dataset, request_by_client=False):
        with torch.no_grad():
            if request_by_client:
                pass
            else:
                if not hasattr(self, "dataset_dict"):
                    self.dataset_dict = {idx: [torch.tensor(df["itemID"].values),
                                               torch.tensor(df["label"].values, dtype=torch.float)]
                                         for idx, df in dataset.df.groupby(["userID"])}
                item_ids = self.dataset_dict[user_id][0]
                target = self.dataset_dict[user_id][1]
            self.batch_size = item_ids.size(0)
            # change to [batch_size, 1]
            item_ids = item_ids.reshape((-1, 1))
            # [batch_size, dim]

            user_embeddings = self.usr(torch.tensor([user_id]))
            entities, relations = self._get_neighbors(item_ids)
            return [self.usr.weight.clone(), entities, self.ent.weight.clone(),
                    relations, self.rel.weight.clone(), target]

    def forward(self, data_batch):
        if len(data_batch) == 1:
            pass
        else:
            user_ids, item_ids = data_batch
            # change to [batch_size, 1]
            self.batch_size = item_ids.size(0)
            item_ids = item_ids.reshape((-1, 1))
            # [batch_size, dim]
            user_ids = user_ids.view((-1, 1))
            user_embeddings = self.usr(user_ids)
            # print('test-----user_embeddings',user_embeddings.shape)
            entities, relations = self._get_neighbors(item_ids)  # 对每一个user-item的item,取item的n_iter层邻居

            entity_set = torch.concat([entity.unique() for entity in entities]).unique()
            entity_map = {int(entity): i for i, entity in enumerate(entity_set)}
            # print('entities1-----------',entities)
            # print('entity_set-----------',entity_set)
            
            entities = [entity.to('cpu') for entity in entities]
            entities = [entity.map_(entity, lambda x, y: entity_map[x]) for entity in entities] 
            # print('entities2-----------',entities)
            entity_embeddings = torch.vstack([self.ent(entity) for entity in entity_set])
            # entity_embeddings = torch.vstack([self.ent(entity) for entity in entities])

            relation_set = torch.concat([relation.unique() for relation in relations]).unique()
            relation_map = {int(relation): i for i, relation in enumerate(relation_set)}
            relations = [relation.to('cpu') for relation in relations]
            relations = [relation.map_(relation, lambda x, y: relation_map[x]) for relation in relations]
            # relations = [relation.to('cpu') for relation in relations]
            relation_embeddings = torch.vstack([self.rel(relation) for relation in relation_set])
            torch.cuda.empty_cache()

            return self.aggregator(user_ids, user_embeddings, entities, entity_embeddings, relations, relation_embeddings)

    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            # 获取每个Item的n_neighbor个邻居， n_iter
            neighbor_entities = self.adj_ent[entities[h]].reshape((self.batch_size, -1))
            neighbor_relations = self.adj_rel[entities[h]].reshape((self.batch_size, -1))
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
            
        # print('_get_neighbors--itemIds', v)
        # print('_get_neighbors--entities',entities)
        # print('_get_neighbors--relations',relations)

        return entities, relations

    def recode_grad(self, flatten_local_models):
        # 梯度聚合：更新主模型（master_model）的梯度信息
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                # print (
                    # f"Before aggregation - Param {i}: Grad mean: {param.grad.mean () if param.grad is not None else 'None'}")
                param.grad=torch.zeros_like(param) # 梯度置零
            for user_id, grad in flatten_local_models.items():
                usr_grad, ent_grad, rel_grad=grad['embeddings_grad']
                model_grad= grad['model_grad']
                 # 打印客户端前梯度
                # print (f"from Client{user_id} - usr_grad mean: {usr_grad.mean ()}")
                # print (f"from Client{user_id} - ent_grad mean: {usr_grad.mean ()}")
                # print (f"from Client{user_id} - rel_grad mean: {rel_grad.mean ()}")
                # print (f"from Client{user_id} - model_grad: {model_grad}")
                # 打印累加前梯度
                # print (f"User {user_id}: Before accumulation - usr.weight.grad mean: {self.usr.weight.grad.mean ()}")
                # print (f"User {user_id}: Before accumulation - ent.weight.grad mean: {self.ent.weight.grad.mean ()}")
                # 嵌入层的权重累加梯度：用户嵌入、实体嵌入、关系嵌入
                self.usr.weight.grad[user_id] += usr_grad[0]
                self.ent.weight.grad[self.id_map[user_id][0]] += ent_grad
                self.rel.weight.grad[self.id_map[user_id][1]] += rel_grad
                # 打印累加后梯度
                # print (f"User {user_id}: After accumulation - usr.weight.grad mean: {self.usr.weight.grad.mean ()}")
                # print (f"User {user_id}: After accumulation - ent.weight.grad mean: {self.ent.weight.grad.mean ()}")
                # 聚合器参数梯度累加
                for i, param in enumerate(self.aggregator.parameters()):
                    # print (f"Aggregator Param {i}: Before accumulation - Grad mean: {param.grad.mean ()}")
                    param.grad += model_grad[i]
                    # print (f"Aggregator Param {i}: After accumulation - Grad mean: {param.grad.mean ()}")
            # 前面累加之后，这里对所有参数求平均
            for param in self.parameters():
                # print (f"Before averaging - Grad mean: {param.grad.mean ()}")
                param.grad/=len(flatten_local_models)
                # print (f"After averaging - Grad mean: {param.grad.mean ()}")
    def recode_grad_by_trainning_num(self, flatten_local_models):
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                param.grad = torch.zeros_like(param)
            totle_interactions = 0
            usr_count = torch.zeros(self.num_usr)
            ent_count = torch.zeros(self.num_ent)
            rel_count = torch.zeros(self.num_rel)
            for user_id, grad in flatten_local_models.items():
                num= len(self.dataset_dict[user_id][0])
                totle_interactions+=num

                model_grad= grad['model_grad']
                usr_grad, ent_grad, rel_grad = grad['embeddings_grad']
                self.usr.weight.grad[user_id] += usr_grad[0]* num
                self.ent.weight.grad[self.id_map[user_id][0]] += ent_grad* num
                self.rel.weight.grad[self.id_map[user_id][1]] += rel_grad* num
                usr_count[user_id]+=num
                ent_count[self.id_map[user_id][0]]+=num
                rel_count[self.id_map[user_id][1]]+=num
                for i,param in enumerate(self.aggregator.parameters()):
                    param.grad+= model_grad[i]*num
            usr_count[usr_count == 0] = 1  # 没有交互的给1，防止零除
            ent_count[ent_count == 0] = 1  # 没有交互的给1，防止零除
            rel_count[rel_count == 0] = 1  # 没有交互的给1，防止零除
            self.usr.weight.grad/= usr_count.reshape([-1, 1])
            self.ent.weight.grad/= ent_count.reshape([-1, 1])
            self.rel.weight.grad/= rel_count.reshape([-1, 1])
            for i, param in enumerate(self.aggregator.parameters()):
                param.grad /= totle_interactions


    def recode_all(self, grads):
        for (usr_grad, ent_grad, rel_grad) in grads:
            if usr_grad != None:
                self.usr_grad += usr_grad
                self.ent_grad += ent_grad
                self.rel_grad += rel_grad


class KGCN_aggregator(torch.nn.Module):
    def __init__(self, batch_size, dim, n_neighbor, aggregator, n_iter):
        super(KGCN_aggregator, self).__init__()
        self.batch_size = None
        self.dim = dim
        self.n_neighbor = n_neighbor
        self.n_iter = n_iter
        self.aggregator = Aggregator(batch_size, dim, aggregator)

    def forward(self, usr_id, usr_embed, ent_id, ent_embed, rel_id, rel_embed):
        # 梯度追踪，设置输入的实体嵌入、关系嵌入和用户嵌入为可计算梯度状态
        ent_embed.requires_grad = True
        rel_embed.requires_grad = True
        usr_embed.requires_grad = True
        self.batch_size = len(ent_id[0])
        if usr_id is None:
            usr_id = [0] * self.batch_size
        # user_embeddings = usr_embed[usr_id] #根据用户 ID 获取对应的用户嵌入
        user_embeddings = usr_embed.squeeze(dim = 1) 
        
        entities_embeddings = [ent_embed[entity] for entity in ent_id] #为每个实体 ID 查找对应的实体嵌入。
        relations_embeddings = [rel_embed[relation] for relation in rel_id] #为每个关系 ID 查找对应的关系嵌入。
        # 调用 _aggregate 方法，将用户、实体和关系的嵌入作为输入，进行多次聚合，生成新的实体嵌入。
        item_embeddings = self._aggregate(user_embeddings, entities_embeddings, relations_embeddings)  # 单层加权求和

        scores = (user_embeddings * item_embeddings).sum(dim=1) #计算评分：计算用户与聚合后的实体嵌入的相似度（点积）
        return torch.sigmoid(scores) #通过 torch.sigmoid 激活函数映射到 [0, 1] 范围内，表示预测的相似度评分

    def _aggregate(self, user_embeddings, entity_embeddings, relation_embeddings):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator.forward(
                    self_vectors=entity_embeddings[hop], #每次迭代生成新的实体嵌入，并将其传递给下一次迭代
                    neighbor_vectors=entity_embeddings[hop + 1].reshape(
                        (self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_embeddings[hop].reshape(
                        (self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_embeddings = entity_vectors_next_iter #entity_embeddings 被更新为每次迭代后的新嵌入。

        return entity_embeddings[0].reshape((self.batch_size, self.dim)) #回最后一次迭代后的实体嵌入


def kgcn(conf):
    if hasattr(conf, "kg"):
        kg, num_user, num_entity, num_relation = conf.kg
    else:
        kg, num_user, num_entity, num_relation = None, 1872, 9366, 60
    device = conf.device
    conf.n_iter = 1
    conf.batch_size = 32
    conf.dim = 16
    conf.neighbor_sample_size = 8
    conf.aggregator = "sum"

    return KGCN_E(num_user, num_entity, num_relation, kg, conf, device)


def kgcn_aggregate(conf):
    return KGCN_aggregator(batch_size=conf.batch_size, dim=conf.dim, n_neighbor=conf.neighbor_sample_size, aggregator=conf.aggregator, n_iter=conf.n_iter)


def kgcn_kg(conf):
    if hasattr(conf, "kg"):
        kg, num_user, num_entity, num_relation = conf.kg
    else:
        kg, num_user, num_entity, num_relation = None, 1872, 9366, 60
    device = conf.device

    return KGCN_kg(num_user, num_entity, num_relation, kg, conf, device)
