# -*- coding: utf-8 -*-
import gc
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.datasets.mixup_data as mixup
import pcode.local_training.compressor as compressor
import pcode.local_training.random_reinit as random_reinit
from pcode.utils.auto_distributed import gather_objects, scatter_objects
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.timer import Timer


class Worker(object):
    def __init__(self, conf):
        # some initializations.
        self.conf = conf
        self.rank = conf.graph.rank
        conf.graph.worker_id = conf.graph.rank #当前进程的唯一标识
        self.conf.device = self.device = torch.device("cuda" if self.conf.graph.on_cuda else "cpu")

        # define the timer for different operations.
        # if we choose the `train_fast` mode, then we will not track the time.
        # 配置conf.train_fast = True， 不跟踪时间
        # 用于在训练过程中跟踪和记录操作的时间
        self.timer = Timer(
            verbosity_level=1 if conf.track_time and not conf.train_fast else 0,
            log_fn=conf.logger.log_metric,
        )

        # create dataset (as well as the potential data_partitioner) for training.
        # dist.barrier()
        # self.dataset = create_dataset.define_dataset(conf, data=conf.data)
        # self.conf.kg = self.dataset["train"].get_kg()
        self.arch = None
        # self.kg = models.__dict__['knowledge_graph'](kg, num_user, num_ent, num_rel).cuda()
        # _, self.data_partitioner = create_dataset.define_data_loader(
        #     self.conf,
        #     dataset=self.dataset["train"],
        #     localdata_id=0,  # random id here.
        #     is_train=True,
        #     data_partitioner=None,
        # )
        conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} initialized the local training data with Master."
        )

        # define the criterion.二进制交叉熵损失函数（BCELoss）
        self.criterion = nn.BCELoss()

        # define the model compression operators.
        # conf.local_model_compression == None
        if conf.local_model_compression is not None:
            if conf.local_model_compression == "quantization":
                self.model_compression_fn = compressor.ModelQuantization(conf)

        self.terminate_batch = 0
        self.last_comm_round = 0

        conf.logger.log(
            f"Worker-{conf.graph.worker_id} initialized dataset/criterion.\n"
        )

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            self._recv_model_from_master()
            self._train()
            self._send_model_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_complete_training():
                return

    def _listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        activation_msg=scatter_objects()[0]
        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = activation_msg['client_id'], \
        activation_msg['comm_round'], activation_msg['local_epoch']
        # self.conf.logger.log (
        #     f"Worker-{self.conf.graph.worker_id} activiated by Master in comm_round--{self.conf.graph.comm_round} epochs--{self.n_local_epochs} ."
        # )

        # once we receive the signal, we init for the local training.
        pass

        # dist.barrier()

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`


        output_list= scatter_objects()[0]

        if output_list != None:
            if output_list['model']:
                self.model=output_list['model']
                self.state_dict= self.model.state_dict()
            else:
                self.model.load_state_dict(self.state_dict)
            # 拆包（unpacking）：*self.input 表示将 output_list['input'] 中的前 n-1 个元素赋值给 self.input
            # entities, relations, targets
            *self.input, self.target = output_list['input']

        # self.conf.logger.log(
        #     f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the model ({output_list['model']}) and embedding from Master."
        # )
        self.metrics = create_metrics.Metrics(self.model, task="null")
        # dist.monitored_barrier()


    def _train(self):
        if self.conf.graph.client_id != -1:
            # KGCN_aggregator 继承自 torch.nn.Module，因此 model.train() 会：
            # 设置 self.training=True。
            # 递归地对所有子模块（如 self.aggregator）也调用 train()，从而切换所有子模块到训练模式。
            self.model.train()

            # init the model and dataloader.
            if self.conf.graph.on_cuda:
                self.model = self.model.to("cuda")
                for i, input in enumerate(self.input):
                    if hasattr(input, "to"):
                        self.input[i]=input.to("cuda")
                    if isinstance(input, list):
                        self.input[i] =[t.to("cuda") if hasattr(t, "to") else t for t in input]
                self.target = self.target.to("cuda")

            # define optimizer, scheduler and runtime tracker.

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr,
                                              weight_decay=self.conf.weight_decay)
            # self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
            # self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
            # self.conf.logger.log(
            #     f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round} n_local_epochs={self.n_local_epochs})."
            # )
            running_loss = 0.0
            for epoch in range(self.n_local_epochs):
                # refresh the logging cache at the end of each epoch.
                self.optimizer.zero_grad()
                # 调用 KGCN_aggregator.forward 方法
                # output 是一个一维张量，大小为 self.batch_size。
                output = self.model(torch.tensor([self.conf.graph.client_id]).to('cuda:0'), *self.input) #前向传播
                # 前向传播（forward）时，input 中的嵌入向量不会改变，但它们会参与计算图构建，允许后续的梯度计算。
                # 后向传播和优化器更新阶段，如果这些嵌入向量是可学习参数，它们的值可能会更新。
                # print(f"对模型输出--{output}和真实标签self.target--{self.target}计算损失.")
                loss = self.criterion(output, self.target) #计算损失
                loss.backward() #反向传播
                # 由于在前向传播中设置了 requires_grad=True，在反向传播时，会计算损失相对于这些嵌入向量的梯度，并存储在 usr_embed.grad, ent_embed.grad, rel_embed.grad 中
                self.optimizer.step ()
                running_loss += loss.item ()
                # print ('[Client {} - total Epoch {}]train_loss: '.format (self.conf.graph.client_id, epoch), loss.item ())
            # print train loss per every local train
            # print ('[Client {} - total Epoch {}]train_loss: '.format (self.conf.graph.client_id, self.n_local_epochs), running_loss/self.n_local_epochs)

            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()
            self.model.to("cpu")

    def print_gpu_tensor(self):
        i = 0
        for obj in gc.get_objects():
            try:
                if (torch.is_tensor(obj) or (
                        hasattr(obj, 'data') and torch.is_tensor(obj.data))) and obj.data.device.type == 'cuda':
                    print(i, ':', type(obj), obj.size())
                    i += 1
            except:
                pass

    def _inference(self, data_batch):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        output = self.model(data_batch["input"])

        # evaluate the output and get the loss, performance.
        if self.conf.use_mixup:
            loss = mixup.mixup_criterion(
                self.criterion,
                output,
                data_batch["target_a"],
                data_batch["target_b"],
                data_batch["mixup_lambda"],
            )

            performance_a = self.metrics.evaluate(loss, output, data_batch["target_a"])
            performance_b = self.metrics.evaluate(loss, output, data_batch["target_b"])
            performance = [
                data_batch["mixup_lambda"] * _a + (1 - data_batch["mixup_lambda"]) * _b
                for _a, _b in zip(performance_a, performance_b)
            ]
        else:
            loss = self.criterion(output, data_batch["target"])
            performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if self.tracker is not None:
            self.tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"][0].size(0)
            )
        return loss, output

    def _add_grad_from_prox_regularized_loss(self):
        assert self.conf.local_prox_term >= 0
        if self.conf.local_prox_term != 0:
            assert self.conf.weight_decay == 0
            assert self.conf.optimizer == "sgd"
            assert self.conf.momentum_factor == 0

            for _param, _init_param in zip(
                    self.model.parameters(), self.init_model.parameters()
            ):
                if _param.grad is not None:
                    _param.grad.data.add_(
                        (_param.data - _init_param.data) * self.conf.local_prox_term
                    )

    def _local_training_with_self_distillation(self, loss, output, data_batch):
        if self.conf.self_distillation > 0:
            loss = loss * (
                    1 - self.conf.self_distillation
            ) + self.conf.self_distillation * self._divergence(
                student_logits=output / self.conf.self_distillation_temperature,
                teacher_logits=self.init_model(data_batch["input"])
                               / self.conf.self_distillation_temperature,
            )
        return loss

    def _divergence(self, student_logits, teacher_logits):
        divergence = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )  # forward KL
        return divergence

    def _turn_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _send_model_to_master(self):
        # dist.monitored_barrier()
        comm_device='cpu'
        if self.conf.graph.client_id != -1:
            gather_dict = {}
            # flatten_model = TensorBuffer(list(self.model.state_dict().values()))
            gather_dict['model_grad']=[param.grad.to(comm_device) for param in self.model.parameters()]

            # self.usr,self.ent,self.rel的梯度
            gather_dict['embeddings_grad']= self.model.get_embed_grad() if self.conf.graph.client_id != -1 else [None] * 3
        else:
            gather_dict=None

        gather_objects(gather_dict)
        # self.conf.logger.log(
        #     f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the model back to Master."
        # )
        # dist.barrier()

    def _terminate_comm_round(self):
        self.model = self.model.cpu()
        # del self.init_model
        self.scheduler.clean()
        self.conf.logger.save_json()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) finished one round of federated learning: (comm_round={self.conf.graph.comm_round})."
        )

    def _terminate_by_early_stopping(self):
        if self.conf.graph.comm_round == -1:
            # dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning by early-stopping."
            )
            return True
        else:
            return False

    def _terminate_by_complete_training(self):
        if self.conf.graph.comm_round == self.conf.n_comm_rounds:
            self.terminate_batch += 1
            if self.terminate_batch == ceil(self.conf.n_participated // self.conf.workers):
                # dist.barrier()
                self.conf.logger.log(
                    f"Worker-{self.conf.graph.worker_id} finished the federated learning: (total comm_rounds={self.conf.graph.comm_round})."
                )
                return True
        else:
            return False

    def _is_finished_one_comm_round(self):
        return True if self.conf.epoch_ >= self.conf.local_n_epochs else False
