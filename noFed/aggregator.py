import torch
import torch.nn.functional as F


class Aggregator(torch.nn.Module):
    '''
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    '''
    
    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator = aggregator
        
    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, act):
        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
        
        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))
            
        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))
            
        else:
            output = neighbors_agg.view((-1, self.dim))
            
        output = self.weights(output)
        return act(output.view((self.batch_size, -1, self.dim)))
        
    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        '''
        This aims to aggregate neighbor vectors
        '''
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))
        
        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim = -1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim = -1)
        
        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim = -1)
        
        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim = 2)
        
        return neighbors_aggregated
    
    
class Aggregator2(torch.nn.Module):
    '''
    Aggregator2
    '''

    def __init__(self, batch_size, dim):
        super(Aggregator2, self).__init__()
        self.batch_size = batch_size
        self.dim = dim  # 16
        self.W1 =  torch.nn.Linear(dim, dim, bias=True)  # 权重矩阵 W1, 输入和输出维度都是dim
        self.W2 =  torch.nn.Linear(1, dim, bias=True)   # 权重矩阵 W2, 输入是点积的标量，所以输入维度是1，输出维度是dim

    def forward(self,item_embeddings, user_embeddings):
        #[1, 16]，[19, 16]
        # print('user_embeddings.shape',user_embeddings.shape)
        # print('item_embeddings.shape',item_embeddings.shape)
        #user_embeddings->[1, dim]   item_embeddings->[batch_size,dim]
        batch_size = item_embeddings.size(0)
        k = 1/item_embeddings.size(1)
       
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # print(batch_size,self.dim)
        # 计算点积：user_embeddings 与 item_embeddings 进行逐个计算
        # user_embeddings = user_embeddings.reshape((self.batch_size, -1))
        # 计算点积
        # print('user_embeddings',user_embeddings,user_embeddings.shape)
        # print('item_embeddings',item_embeddings,item_embeddings.shape)
        # [batch_size]
        dot_product = (user_embeddings * item_embeddings).sum(dim=-1)  # shape: [19]
        # print('dot_product',dot_product,dot_product.shape)

        # [batch_size,dim]
        b = self.W2(dot_product.unsqueeze(1) )  # shape: [19, 1]->[19, dim]    
        # 计算权重矩阵 W1 变换后的用户嵌入 (作用于user_embeddings)
        user_masg = self.W1(user_embeddings)  # shape: [dim]
        a = self.W1(item_embeddings)  # shape: [19, 16]
        # print('a',a,a.shape)
        # print('b',b,b.shape)

        # 对每个项目的 a 和 b 求和
        # [batch_size,dim]
        result = (a + b)*k # shape: [19, 16]
        # print('result',result,result.shape)
        result = result.sum(dim=0)
        # print('sum_result',result,result.shape)
        # 最终通过 Sigmoid 函数
        output = result + user_masg  # 将用户的嵌入加到输出中
        # print('user_masg',user_masg,user_masg.shape)
        # print('output',output,output.shape)
        # print('output', torch.tanh(output), torch.tanh(output).shape)
        # breakpoint()

        # print("Output:", output)  # 输出最终结果
        # print("input:", user_embeddings1)  # 输出最终结果
        return  torch.tanh(output)