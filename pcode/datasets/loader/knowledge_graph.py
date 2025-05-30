import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
import os


class RecommendationDS(data.Dataset):
    '''
    Data Loader class which makes dataset for training / knowledge graph dictionary
    '''

    def __init__(self, data, train=True):
        self.preprocessed_data_path = Path(f'./data/{data}/preprocessed_data')
        if not self.preprocessed_data_path.exists():  # music

            self.cfg = {
               'movie': {
                    'item2id_path': 'data/movie/item_index2entity_id.txt',
                    'kg_path': 'data/movie/kg_final.txt',
                    'rating_path': 'data/movie/ratings_final.txt',
                    'rating_sep': ',',
                    'threshold': 4.0
                },
                'music': {
                    'item2id_path': 'data/music/item_index2entity_id.txt',
                    'kg_path': 'data/music/kg_final.txt',
                    'rating_path': 'data/music/ratings_final.txt',
                    'rating_sep': '\t',
                    'threshold': 0.0
                },
                'book': {
                    'item2id_path': 'data/book/item_index2entity_id_rehashed.txt',
                    'kg_path': 'data/book/kg_rehashed.txt',
                    'rating_path': 'data/BX-Book-Ratings.csv',
                    'rating_sep': ';',
                    'threshold': 0.0
                }
            }
            self.data = data
            self.test_ratio = 0.2
            # # 数据加载和预处理
            df_item2id = pd.read_csv(self.cfg[data]['item2id_path'], sep='\t', header=None, names=['item', 'id'])
            df_kg = pd.read_csv(self.cfg[data]['kg_path'], sep='\t', header=None, names=['head', 'relation', 'tail'])
            df_rating = pd.read_csv(self.cfg[data]['rating_path'], sep=self.cfg[data]['rating_sep'],
                                    names=['userID', 'itemID', 'rating'],skiprows=1)
            # print(df_rating['userID'].nunique())
            # df_rating['itemID'] and df_item2id['item'] both represents old entity ID
            # 数据清洗与处理
            df_rating = df_rating[df_rating['itemID'].isin(df_item2id['item'])]  # 只取item2id里存在的item
            df_rating= df_rating.groupby('userID').filter(lambda x: len(x) > 10)  # 只保留那些评分项目数大于 10 的用户，保证每个用户有足够的评分数据进行训练
            df_rating.reset_index(inplace=True, drop=True)
            
            
            self.df_item2id = df_item2id  # item和id的对应关系
            self.df_kg = df_kg  # 知识图谱
            self.df_rating = df_rating  # 只包含对应关系item的购买记录 pd(user,item,rating)
            # print('self.df_rating')
            # print(self.df_rating.info())
            # print( self.df_rating["userID"].apply(type).unique())
            # print(self.df_rating['userID'].nunique())

            self.user_encoder = LabelEncoder()# 拟合用户 ID 列，将所有用户的 ID 转换为整数编码。
            self.entity_encoder = LabelEncoder()
            self.relation_encoder = LabelEncoder()
            
            self._encoding()
            
            kg = self._construct_kg()  # {head: (relation,tails)} 无向图，正反同关系
            df_dataset = self._build_dataset()
            # print('df_dataset')
            # print(df_dataset.head())
            # print(df_dataset["userID"].apply(type).unique())
            train_set, test_set, _, _ = train_test_split(df_dataset, df_dataset['label'], test_size=self.test_ratio,
                                                         shuffle=False, random_state=999)
            # print('train_set')
            # print(train_set.head())
            # print(train_set["userID"].apply(type).unique())
            num_user, num_entity, num_relation = self.get_num()

            # 确保训练集和测试集中的用户 ID 一致。如果测试集包含训练集中没有的用户 ID，则将这些用户的记录移动到训练集中。
            train_userIDs = set(train_set['userID'])
            test_userIDs = set(test_set['userID'])
            userIDs_to_move = test_userIDs - train_userIDs
            rows_to_move = test_set[test_set['userID'].isin(userIDs_to_move)]
            # train_set = train_set.append(rows_to_move)
            train_set= pd.concat ([train_set, rows_to_move],ignore_index=True)
            test_set = test_set[~test_set['userID'].isin(userIDs_to_move)]
            # print('train_set22222')
            # print(train_set.head())
            # print(train_set["userID"].apply(type).unique())
            print('num_user ', num_user)

            print ('--------------------num_user ', num_user)
            print ( '---------------------num_entity ',num_entity)
            print ( '---------------------num_relation ',num_relation)
            
            torch.save(
                {'train_set': train_set, 'test_set': test_set, 'kg': kg, 'num_user': num_user, 'num_entity': num_entity,
                 'num_relation': num_relation},
                self.preprocessed_data_path)
        else:
            preprocessed_data = torch.load(self.preprocessed_data_path)
            train_set, test_set = preprocessed_data['train_set'], preprocessed_data['test_set']


        # breakpoint()
        # 为每个用户生成索引，便于快速查找该用户对应的评分记录
        self.df = train_set if train else test_set
        self.idx = self.df.index
        self.index = defaultdict(list)
        # self.user_num = self.df.userID.max() + 1
        # print(self.df.head())
        # print(self.df[self.df['userID'] == 378])
        # print(self.df.info())
        # print(self.df["userID"].apply(type).unique())  # 查看所有数据类型
        
        # print(self.df["userID"].unique())  # 查看所有唯一值
        # print(self.df["userID"].max ())
        self.user_num = self.df["userID"].max () + 1
        for user_id in range(self.user_num):
            self.index[user_id] = self.df[self.df.userID == user_id].index

    def _encoding(self):
        '''
        Fit each label encoder and encode knowledge graph
        '''
        self.user_encoder.fit(self.df_rating['userID']) # 拟合用户 ID 列，将所有用户的 ID 转换为整数编码。
        # df_item2id['id'] and df_kg[['head', 'tail']] represents new entity ID
        self.entity_encoder.fit(
            pd.concat([self.df_kg['head'], self.df_kg['tail']]))  # id是item的紧凑表示, 必定在entitiy里

        self.relation_encoder.fit(self.df_kg['relation'])

        # encode df_kg通过 transform 方法将实际的实体 ID 或关系数据转换为对应的编码值
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])

    def _build_dataset(self):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        # df_rating update
        df_dataset = pd.DataFrame()
        df_dataset['userID'] = self.user_encoder.transform(self.df_rating['userID'])
       
        # update to new id  映射项目 ID：将原始的项目（item）映射到新的项目 ID（id）。
        item2id_dict = dict(zip(self.df_item2id['item'], self.df_item2id['id']))
        # 问题？item2id_dict[x]还是原来的id啊，并不是实体id,应该是将item名换成itemiD
        self.df_rating['itemID'] = self.df_rating['itemID'].apply(lambda x: item2id_dict[x])  # item 映射为 entity id
        df_dataset['itemID'] = self.entity_encoder.transform(self.df_rating['itemID'])  # 紧凑表示

        df_dataset['label'] = self.df_rating['rating'].apply(
            lambda x: 0 if x < self.cfg[self.data]['threshold'] else 1)  # label二值化[0,1]

        # negative sampling负样本采样
        df_dataset = df_dataset[df_dataset['label'] == 1]  # 只取rating大于阈值的正样本
        # df_dataset requires columns to have new entity ID
        df_dataset['userID'] = df_dataset['userID'].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        full_item_set = set(range(len(self.entity_encoder.classes_)))
        user_list = []
        item_list = []
        label_list = []
        df_dataset = df_dataset.reset_index(drop=True)  # 重置索引，丢弃原来的索引
        for user, group in df_dataset.groupby(df_dataset['userID']):
            #  对每个用户（user），计算该用户的正样本项目集合（item_set）。
            item_set = set(group['itemID'])
            negative_set = full_item_set - item_set # 是去除用户已评分项目后的所有项目，即潜在的负样本
            negative_sampled = random.sample(negative_set, len(item_set))  # 从 negative_set 中随机采样与该用户的正样本数量相等的负样本。
            user_list.extend([user] * len(negative_sampled))
            item_list.extend(negative_sampled)
            label_list.extend([0] * len(negative_sampled))
        negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list})  # 负样本label为0
        print('negative build_dataset')
        print(negative.head())
        print(negative["userID"].apply(type).unique())
        df_dataset = pd.concat([df_dataset, negative]) # 通过 pd.concat 将正样本和负样本合并成一个完整的训练数据集
        

        df_dataset = df_dataset.sample(frac=1, replace=False, random_state=999)  # 不放回抽样1,相当于打乱
        df_dataset.reset_index(inplace=True, drop=True) # 重新设置数据集的索引
        print('Done')
        print('df_dataset build_dataset')
        print(negative.head())
        print(negative["userID"].apply(type).unique())
        # df_dataset是pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list}) 形式
        return df_dataset

    def _construct_kg(self):
        '''
        Construct knowledge graph
        Knowledge graph is dictionary form
        'head': [(relation, tail), ...]
        '''
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for i in range(len(self.df_kg)):
            head = self.df_kg.iloc[i]['head']
            relation = self.df_kg.iloc[i]['relation']
            tail = self.df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        print('Done')
        return kg

    def get_kg(self):
        preprocessed_data = torch.load(self.preprocessed_data_path)
        return preprocessed_data['kg'], preprocessed_data['num_user'], preprocessed_data['num_entity'], \
        preprocessed_data['num_relation']

    def get_encoders(self):
        return (self.user_encoder, self.entity_encoder, self.relation_encoder)

    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        user_id = np.array(self.df.loc[self.idx[idx]]['userID'])
        item_id = np.array(self.df.loc[self.idx[idx]]['itemID'])
        label = np.array(self.df.loc[self.idx[idx]]['label'], dtype=np.float32)
        return (user_id, item_id), label

    def set_user(self, user_id):
        if isinstance(user_id, int):
            if user_id == -1:
                self.idx = self.df.index
            else:
                self.idx = self.index[user_id]
        else:
            self.idx = self.index[user_id[0]]
            for id in user_id[1:]:
                self.idx = self.idx.append(self.index[id])

        return self

    def get(self, attr):
        # 用于获取当前 self.idx 所对应的用户数据中特定属性（列）的值。
        return self.df.loc[self.idx][attr].values
