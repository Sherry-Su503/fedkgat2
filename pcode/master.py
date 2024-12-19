# -*- coding: utf-8 -*-
import collections
import copy
import os
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import wandb

import pcode.create_aggregator as create_aggregator
import pcode.create_coordinator as create_coordinator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.master_utils as master_utils
import pcode.utils.checkpoint as checkpoint
from pcode.utils.auto_distributed import gather_objects, scatter_objects
from pcode.utils.early_stopping import EarlyStoppingTracker
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.topk_eval import TopkEval
from wandb_utils import init_wandb


class Master(object):
    def __init__(self, conf):
        # 结果可视化
        init_wandb(conf, 'graph_recommendation', conf.experiment,
                   {"loss": "min", "accuracy": "max", "auc": "max", "precision": "max", "recall": "max", "ndcg": "max"},
                   "comm_round")

        self.init_parameters(conf)

        # dist.barrier()
        # 加载数据
        self.init_dataloader(conf)

        # 初始化模型
        self.init_model(conf)

        # 用于初始化损失函数（criterion）和评估指标（metrics）
        self.init_criterion_and_metric(conf)

        # define the aggregators.聚合器
        self.aggregator = create_aggregator.Aggregator(
            conf,
            model=self.master_model,
            criterion=self.criterion,
            metrics=self.metrics,
            dataset=self.dataset,
            test_loaders=self.test_loaders,
            clientid2arch=self.clientid2arch,
        )

        # define early_stopping_tracker.
        # conf.early_stopping_rounds ===0
        # conf.early_stopping_rounds 耐心轮次（例如：如果设置为 5，则表示模型在验证集上连续 5 次性能没有提升，训练就会提前结束）
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )

        # save arguments to disk.
        conf.is_finished = False
        # 调用 checkpoint 对象的 save_arguments 方法。作用是将配置参数保存到磁盘上
        checkpoint.save_arguments(conf)

        # simulation parameter
        self.worker_archs = collections.defaultdict(str) #默认值为 str（即空字符串）
        self.last_comm_round = 0 #用于跟踪训练过程中的最后一个通信轮次

    def init_criterion_and_metric(self, conf):

        # define the criterion and metrics.
        # 用于初始化损失函数（criterion）和评估指标（metrics）
        # self.criterion = cross_entropy.CrossEntropyLoss(reduction="mean")
        # 定义评估指标
        # TopkEval 的构造函数中传入了数据配置 (self.conf.data)、训练数据集 (self.dataset['train'])、测试数据集 (self.dataset['test'])，以及 k_list（表示不同的 k 值）。
        self.topk_eval = TopkEval(self.conf.data, self.dataset['train'], self.dataset['test'],k_list=self.conf.k_list)
        # 定义损失函数，一个二元交叉熵损失函数，常用于二分类任务。
        self.criterion = torch.nn.BCELoss()
        # 定义评估指标
        self.metrics = create_metrics.Metrics(self.master_model, task="recommondation")
        # 记录一条日志，表示模型、数据集、损失函数和评估指标初始化完成
        conf.logger.log(f"Master initialized model/dataset/criterion/metrics.")
        # 初始化协调器
        self.coordinator = create_coordinator.Coordinator(conf, self.metrics)
        # 记录日志，表示聚合器或协调器已初始化完成
        conf.logger.log(f"Master initialized the aggregator/coordinator.\n")

    def init_dataloader(self, conf):
        # 初始化训练、验证和测试数据的加载器，并支持按用户分区的加载器设置，用于不同的客户端数据拆分
        # 该函数会返回一个字典格式的数据集对象 self.dataset，可能包含训练、验证和测试数据集
        self.dataset = create_dataset.define_dataset(conf, data=conf.data)
        # _, self.data_partitioner = create_dataset.define_data_loader(
        #     self.conf,
        #     dataset=self.dataset["train"],
        #     localdata_id=0,  # random id here.
        #     is_train=True,
        #     data_partitioner=None,
        # )
        # 记录日志，表示主进程已初始化本地训练数据。
        conf.logger.log(f"Master initialized the local training data with workers.")
        # create val loader.初始化验证集加载器
        # right now we just ignore the case of partitioned_by_user.
        # 检查是否存在验证集 self.dataset["val"]
        if self.dataset["val"] is not None:
            assert not conf.partitioned_by_user
            self.val_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["val"], is_train=False
            )
            conf.logger.log(f"Master initialized val data.")
        else:
            self.val_loader = None
        # create test loaders.初始化测试集加载器
        # localdata_id start from 0 to the # of clients - 1. client_id starts from 1 to the # of clients.
        # 根据配置 conf.partitioned_by_user 判断是否按用户分区
        if conf.partitioned_by_user:
            # 如果按用户分区，将每个客户端的数据集创建成一个单独的测试加载器，存入 self.test_loaders 列表中
            self.test_loaders = []
            for localdata_id in self.client_ids:
                test_loader, _ = create_dataset.define_data_loader(
                    conf,
                    self.dataset["test"],
                    localdata_id=localdata_id,
                    is_train=False,
                    shuffle=False,
                )
                self.test_loaders.append(copy.deepcopy(test_loader))
        else:
            # 直接创建一个单一的测试加载器 test_loader
            test_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["test"], is_train=False
            )
            print('self.dataset["test"]',self.dataset["test"])
            self.test_loaders = [test_loader]

    def init_parameters(self, conf):
        # some initializations.
        # 对参数进行初始化，包括配置对象、客户端和参与节点的 ID 列表，以及设备选择
        self.conf = conf
        # 初始化客户端 ID 列表:使用conf.n_clients（客户端数量）创建一个从 0 到 n_clients - 1 的 ID 列表，
        self.client_ids = list(range(conf.n_clients))
        # 初始化（表示每回合参与的客户端数目）参与节点 ID 列表:使用 conf.n_participated（参与节点数量）创建一个从 1 到 n_participated 的 ID 列表，赋值给 self.world_ids
        self.world_ids = list(range(1, 1 + conf.n_participated))
        # 选择计算设备
        self.conf.device = self.device = torch.device("cuda" if conf.on_cuda else "cpu")

    def init_model(self, conf):
        # 用于初始化模型及其状态字典，包括主模型、客户端模型架构，以及客户端 ID 与架构的映射关系
        # create model as well as their corresponding state_dicts.
        # 获取知识图
        self.conf.kg = self.dataset["train"].get_kg()
        print('def init_model(self, conf)---------------------------------------')
        print(self.dataset["train"])
        self.conf.n_clients = self.conf.kg[1]
        print('self.conf.n_clients-----------------------------------------------')
        print(self.conf.n_clients)
        # 初始化主模型
        _, self.master_model = create_model.define_model(
            conf, to_consistent_model=False
        )
        # 确定客户端架构:所有客户端的架构被放入一个集合中。如果多个客户端使用相同的架构，则该架构只会在集合中出现一次。
        self.used_client_archs = set(
            [
                create_model.determine_arch(conf, client_id, use_complex_arch=True)
                for client_id in range(1, 1 + conf.n_clients)
            ]
        )
        self.conf.used_client_archs = self.used_client_archs
        # conf.logger.log(f"The client will use archs={self.used_client_archs}.")
        # conf.logger.log("Master created model templates for client models.")


        # 初始化客户端模型
        self.client_models = dict(
            create_model.define_model(conf, to_consistent_model=False, arch=arch)
            for arch in self.used_client_archs
        )
        # 将第一个客户端模型的状态字典加载到主模型的聚合器 aggregator 中，以确保模型参数的一致性
        # state_dict() 是 PyTorch 中的一个方法，用于返回模型的状态字典，其中包含了模型的所有参数（权重和偏置）。
        # load_state_dict() 是 PyTorch 中的方法，用于将一个字典中的参数加载到模型中。它会更新模型中所有与字典中键匹配的参数。
        # self.master_model.aggregator.load_state_dict(list(self.client_models.values())[0].state_dict())
        # 初始化主模型的梯度为零，避免梯度的累积
        # .parameters() 是 PyTorch 模型中的一个方法，它返回模型的所有可训练参数（如权重、偏置等）。每个参数都是一个 torch.Tensor 对象
        for param in self.master_model.parameters():
            param.grad=torch.zeros_like(param) #torch.zeros_like(param) 返回一个与 param 同形状的零张量

        # 创建客户端 ID 与架构的映射
        self.clientid2arch = dict(
            (
                client_id,
                create_model.determine_arch(
                    conf, client_id=client_id, use_complex_arch=True
                ),
            )
            for client_id in range(conf.n_clients)
        )
        print('conf.n_clients',conf.n_clients)
        self.conf.clientid2arch = self.clientid2arch
        # conf.logger.log(
        #     f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
        # )

    def run(self):
        # run 方法实现了一个用于联邦学习的循环过程，主要包括通信轮次的管理、客户端选择、模型传递、聚合、早停检测等操作。
        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            # 循环遍历 comm_round（通信轮次）
            self.conf.graph.comm_round = comm_round
            self.conf.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # get random n_local_epochs.获取随机的本地训练轮次数
            list_of_local_n_epochs = self._get_n_local_epoch(
                conf=self.conf, n_participated=self.conf.n_participated
            )
            # random select clients from a pool.客户端池中随机选择一些客户端参与当前通信轮次
            selected_client_ids = self._random_select_clients()
            selected_client_ids, list_of_local_n_epochs = self._select_clients_per_round(selected_client_ids,
                                                                                         list_of_local_n_epochs)

            # start one comm_round, parallel worker number of threads
            # 用于存储客户端上传的本地模型
            flatten_local_models={}
            for i, (client_ids, local_n_epoch) in enumerate(zip(selected_client_ids, list_of_local_n_epochs)):

                # detect early stopping.检测是否满足早停条件，如果满足则提前结束训练
                # self._check_early_stopping()
                # get current workers' client id.

                # init the activation tensor and broadcast to all clients (either start or stop).激活当前选定的客户端
                self._activate_selected_clients(
                    client_ids, self.conf.graph.comm_round, local_n_epoch
                )

                # will decide to send the model or stop the training.
                if not self.conf.is_finished:
                    # broadcast the model to activated clients.
                    self._send_model_to_selected_clients(client_ids)
                else:
                    # dist.barrier()
                    self.conf.logger.log(
                        f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                    )
                    return

                # wait to receive the local models.
                flatten_local_models.update(self._receive_models_from_selected_clients(
                    client_ids
                ))

            # aggregate the local models and evaluate on the validation dataset.
            self._aggregate(flatten_local_models)
            self._evaluate()

            # evaluate the aggregated model.
            self.conf.logger.log(f"Master finished one round of federated learning.\n")

        # formally stop the training (the master has finished all communication rounds).
        # dist.barrier()
        self._finishing()

    def _select_clients_per_round(self, selected_client_ids, list_of_local_n_epochs):
        # 将参与训练的客户端分配给
        client_ids = []
        local_n_epochs = []
        #  self.conf.workers进程总数
        if self.conf.n_participated % self.conf.workers:
            selected_client_ids += [-1] * (self.conf.workers - self.conf.n_participated % self.conf.workers)
            list_of_local_n_epochs += [-1] * (self.conf.workers - self.conf.n_participated % self.conf.workers)
        for i in range(0, self.conf.n_participated, self.conf.workers):
            client_ids.append(selected_client_ids[i:i + self.conf.workers])
            local_n_epochs.append(list_of_local_n_epochs[i:i + self.conf.workers])

        return client_ids, local_n_epochs

    def _random_select_clients(self):
        ''' 从n_clients中随机选择n_participated个客户端参与本轮次训练'''
        selected_client_ids = self.conf.random_state.choice(
            self.client_ids, self.conf.n_participated, replace=False
        ).tolist()
        selected_client_ids.sort()
        self.conf.logger.log(
            f"Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
        )
        # if len(selected_client_ids)% self.conf.workers!=0:
        #     selected_client_ids+= [-1]*(self.conf.workers-len(selected_client_ids) % self.conf.workers)
        return selected_client_ids

    def _activate_selected_clients(
            self, selected_client_ids, comm_round, list_of_local_n_epochs
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        scatter_list=[]
        for selected_client_id, local_n_epoch in zip( selected_client_ids ,list_of_local_n_epochs):
            activation_msg={}
            activation_msg['client_id']=selected_client_id
            activation_msg['comm_round'] =comm_round
            activation_msg['local_epoch'] =local_n_epoch
            # client_arch= self.clientid2arch[selected_client_id]
            # rank=dist.get_rank()
            # if selected_client_id != -1 and  self.worker_archs[rank]!= client_arch :
            #     model = copy.deepcopy(self.client_models[client_arch])
            #     activation_msg.append(model)
            # else:
            #     activation_msg.append(None)
            scatter_list.append(activation_msg)
        scatter_objects(scatter_list)
        self.conf.logger.log(f"Master activated the selected clients.")
        # dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        self.conf.logger.log(f"Master send the models to workers.")
        scatter_list = []
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            # transfer parameters if new comm_round and client arch not changed.
            distribut_dict = {}
            if selected_client_id != -1:
                # 只下发，模型的参数
                distribut_dict['model'] = self.master_model.state_dict()
                # item_ids, targets = self.master_model._get_items(selected_client_id, self.dataset["train"], self.conf.local_batch_size)
                # distribut_dict['input'] = [item_ids,targets]
                scatter_list.append(distribut_dict)
            else:
                scatter_list.append(None)
            # self.conf.logger.log (f"\tMaster send the current model={client_arch} to process_id={worker_rank}")
        scatter_objects(scatter_list)

        self.last_comm_round = self.conf.graph.comm_round
        self.conf.logger.log(
            f"\tMaster send the current model to client."
        )
        # dist.monitored_barrier()


    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local models.")
        # dist.monitored_barrier()
        output = gather_objects()
        # outout[0]是主节点的None
        flatten_local_models={client:model for client,model in zip(selected_client_ids, output[1:]) if client!=-1}
        # dist.barrier()
        self.conf.logger.log(f"Master received all local models.")
        return flatten_local_models


    def _avg_over_archs(self, flatten_local_models):
        # get unique arch from this comm. round.
        archs = set(
            [
                self.clientid2arch[client_idx]
                for client_idx in flatten_local_models.keys()
            ]
        )

        # average for each arch.
        archs_fedavg_models = {}
        for arch in archs:
            # extract local_models from flatten_local_models.
            _flatten_local_models = {}
            for client_idx, flatten_local_model in flatten_local_models.items():
                if self.clientid2arch[client_idx] == arch:
                    _flatten_local_models[client_idx] = flatten_local_model

            # average corresponding local models.
            self.conf.logger.log(
                f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
            )
            fedavg_model = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                flatten_local_models=_flatten_local_models,
                aggregate_fn_name="_s1_federated_average",
            )
            archs_fedavg_models[arch] = fedavg_model
        return archs_fedavg_models

    def _aggregate(self, flatten_local_models):
        # uniformly averaged the model before the potential aggregation scheme.
        # same_arch = (
        #         len(self.client_models) == 1
        #         and self.conf.arch_info["master"] == self.conf.arch_info["worker"][0]
        # )
        # conf.same_arch=True
        same_arch = self.conf.same_arch
        # uniformly average local models with the same architecture.
        # fedavg_models = self._avg_over_archs(flatten_local_models)

        # 主模型更新梯度
        # self.master_model.recode_grad(flatten_local_models)
        self.master_model.recode_grad_by_trainning_num (flatten_local_models)

        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self.master_model.parameters(), lr=self.conf.lr,
                                              weight_decay=self.conf.weight_decay)
        if same_arch:
            # TODO: 如何处理grad
            # 打印参数更新前的状态
            # for i, param in enumerate (self.master_model.parameters ()):
            #     print (
            #         f"Before optimizer step - Param {i}: Value mean: {param.data.mean ()} Grad mean: {param.grad.mean ()}")
            self.optimizer.step() # 根据累加后的梯度，更新主模型的参数
            # for i, param in enumerate (self.master_model.parameters ()):
            #     print (f"After optimizer step - Param {i}: Value mean: {param.data.mean ()}")
            self.optimizer.zero_grad(set_to_none=False)# 清空梯度
            # fedavg_model = copy.deepcopy(self.master_model.aggregator) #深拷贝主模型的聚合器
            # fedavg_models = {'kgcn_aggregate': fedavg_model}
            # # fedavg_model = list(fedavg_models.values())[0]
        else:
            fedavg_model = None

#         # (smarter) aggregate the model from clients.
#         # note that: if conf.fl_aggregate["scheme"] == "federated_average",
#         #            then self.aggregator.aggregate_fn = None.
#         if self.aggregator.aggregate_fn is not None:
#             # evaluate the uniformly averaged model.
#             if fedavg_model is not None:
#                 performance = master_utils.get_avg_perf_on_dataloaders(
#                     self.conf,
#                     self.coordinator,
#                     fedavg_model,
#                     self.criterion,
#                     self.metrics,
#                     self.test_loaders,
#                     label=f"fedag_test_loader",
#                 )
#             else:
#                 assert "knowledge_transfer" in self.conf.fl_aggregate["scheme"]

#                 performance = None
#                 for _arch, _fedavg_model in fedavg_models.items():
#                     master_utils.get_avg_perf_on_dataloaders(
#                         self.conf,
#                         self.coordinator,
#                         _fedavg_model,
#                         self.criterion,
#                         self.metrics,
#                         self.test_loaders,
#                         label=f"fedag_test_loader_{_arch}",
#                     )

#             # aggregate the local models.
#             client_models = self.aggregator.aggregate(
#                 master_model=self.master_model,
#                 client_models=self.client_models,
#                 fedavg_model=fedavg_model,
#                 fedavg_models=fedavg_models,
#                 flatten_local_models=flatten_local_models,
#                 performance=performance,
#             )
#             # here the 'client_models' are updated in-place.
#             if same_arch:
#                 # here the 'master_model' is updated in-place only for 'same_arch is True'.
#                 self.master_model.load_state_dict(
#                     list(client_models.values())[0].state_dict()
#                 )
#             for arch, _client_model in client_models.items():
#                 self.client_models[arch].load_state_dict(_client_model.state_dict())
#         else:
#             # update self.master_model in place.
#             # if same_arch:
#             #     self.master_model.load_state_dict(fedavg_model.state_dict())
#             # update self.client_models in place.
#             # 更新客户端模型参数
#             for arch, _fedavg_model in fedavg_models.items():
#                 # 表示将 _fedavg_model 的权重和偏置参数加载到 self.client_models[arch] 中的对应模型
#                 self.client_models[arch].load_state_dict(_fedavg_model.state_dict())
        self.conf.logger.log (
            f"\tMaster finish aggregate the models."
        )


    def _evaluate(self):
        # 每隔validation_interval轮，验证一次模型性能，保存最佳性能
        if self.conf.graph.comm_round % self.conf.validation_interval == 0:
            self._validation()
        # 每隔topk_eval_interval轮，评估一次性能
        if self.conf.graph.comm_round % self.conf.topk_eval_interval == 0:
            self.topk_eval.eval(self.master_model, self.last_comm_round)


    def _validation(self):
        if self.conf.same_arch:
            master_utils.do_validation(
                self.conf,
                self.coordinator,
                self.master_model,
                self.criterion,
                self.metrics,
                self.test_loaders,
                label=f"aggregated_test_loader",
            )
        else:
            for arch, _client_model in self.client_models.items():
                master_utils.do_validation(
                    self.conf,
                    self.coordinator,
                    _client_model,
                    self.criterion,
                    self.metrics,
                    self.test_loaders,
                    label=f"aggregated_test_loader_{arch}",
                )

    def _check_early_stopping(self):
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                    self.coordinator.key_metric.cur_perf is not None
                    and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.conf.logger.log("Master early stopping: meet target perf.")
                self.conf.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.conf.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True
        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.conf.graph.comm_round - 1
            self.conf.graph.comm_round = -1
            self._finishing(_comm_round)

    def _finishing(self, _comm_round=None):
        self.conf.logger.save_json()
        self.conf.logger.log(f"Master finished the federated learning.")
        self.conf.is_finished = True
        self.conf.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")


    def _get_n_local_epoch(self, conf, n_participated):
        #  conf.min_local_epochs ===None
        if conf.min_local_epochs is None:
            # local_n_epochs == 1, n_participated===32
            return [conf.local_n_epochs] * n_participated
        else:
            # here we only consider to (uniformly) randomly sample the local epochs.
            assert conf.min_local_epochs > 1.0
            random_local_n_epochs = conf.random_state.uniform(
                low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
            )
            return random_local_n_epochs
