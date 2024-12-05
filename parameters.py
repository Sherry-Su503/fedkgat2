# -*- coding: utf-8 -*-
"""define all global parameters here."""
import argparse
import os
import random
import socket
import time
import traceback
from contextlib import closing
from os.path import join
from pathlib import Path

import numpy as np
import pynvml
import torch
import torch.distributed as dist
import yaml

import pcode.models as models
from pcode.utils import topology, param_parser, checkpoint, logging
from pcode.utils.param_parser import str2bool

def get_args():
    parser = add_argument()
    conf=parse_args_with_yaml(parser)
    set_environ()
    debug_parameter(conf)
    complete_missing_config(conf)
    experiment_paramenter(conf)
    validate_config(conf)
    return conf


def add_argument():
    ROOT_DIRECTORY = "./"
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, "data/")
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, "checkpoint")
    model_names = sorted(
        name for name in models.__dict__ if name.islower() and not name.startswith("__")
    )
    # feed them to the parser.创建了一个 ArgumentParser 对象 parser
    parser = argparse.ArgumentParser(description="PyTorch Training for ConvNet")
    # add arguments.
    parser.add_argument("--work_dir", default=None, type=str)
    parser.add_argument("--remote_exec", default=False, type=str2bool)

    # dataset.数据集参数
    parser.add_argument("--data", default="cifar10", help="a specific dataset name")
    parser.add_argument("--val_data_ratio", type=float, default=0)
    parser.add_argument(
        "--train_data_ratio", type=float, default=0, help="after the train/val split."
    )
    parser.add_argument(
        "--data_dir", default=RAW_DATA_DIRECTORY, help="path to dataset"
    )
    parser.add_argument("--img_resolution", type=int, default=None)
    parser.add_argument("--use_fake_centering", type=str2bool, default=False)
    parser.add_argument(
        "--use_lmdb_data",
        default=False,
        type=str2bool,
        help="use sequential lmdb dataset for better loading.",
    )
    parser.add_argument(
        "--partition_data",
        default=None,
        type=str,
        help="decide if each worker will access to all data.",
    )
    parser.add_argument("--pin_memory", default=True, type=str2bool)
    parser.add_argument(
        "-j",
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    # 是否使用均值和标准差来对输入数据进行标准化处理
    parser.add_argument(
        "--pn_normalize", default=True, type=str2bool, help="normalize by mean/std."
    )

    # ---model参数
    parser.add_argument(
        "--arch",
        default="resnet20",
        help="model architecture: " + " | ".join(model_names) + " (default: resnet20)",
    )
    # 用于指定组归一化（Group Normalization）的组数
    parser.add_argument("--group_norm_num_groups", default=None, type=int)
    # 当主模型和工作模型不同（例如，在分布式训练中），此参数可以用来指定主模型（master）和工作模型（worker）的架构。默认值为 master=resnet20,worker=resnet8:resnet14，这表示主模型使用 resnet20，工作模型则使用 resnet8 和 resnet14。
    parser.add_argument(
        "--complex_arch", type=str, default="master=resnet20,worker=resnet8:resnet14",
        help="specify the model when master and worker are not the same",
    )
    # 是否使用卷积层的偏置项
    parser.add_argument("--w_conv_bias", default=False, type=str2bool)
    # 是否使用全连接层的偏置项
    parser.add_argument("--w_fc_bias", default=True, type=str2bool)
    # 是否冻结批量归一化层（Batch Normalization, BN
    parser.add_argument("--freeze_bn", default=False, type=str2bool)
    # 是否冻结批量归一化的仿射参数
    parser.add_argument("--freeze_bn_affine", default=False, type=str2bool)
    # 用于调整 ResNet 网络架构的缩放因子
    parser.add_argument("--resnet_scaling", default=1, type=float)
    # 用于调整 VGG 网络架构的缩放因子。
    parser.add_argument("--vgg_scaling", default=None, type=int)
    # 用于指定 Evonorm 方法的版本。如果使用 Evonorm 进行归一化或其他操作，可以通过此参数设置版本号
    parser.add_argument("--evonorm_version", default=None, type=str)

    # ---联邦训练参数
    # data, training and learning scheme.
    # 指定联邦学习中的通信回合数（即通信的次数
    parser.add_argument("--n_comm_rounds", type=int, default=90)
    # 指定目标性能值，通常用于设置目标精度或损失值，
    parser.add_argument(
        "--target_perf", type=float, default=None, help="it is between [0, 100]."
    )
    # 用于设置在达到早期停止条件时的最大回合数。如果模型在连续多个回合中没有性能提升，训练将提前停止。
    parser.add_argument("--early_stopping_rounds", type=int, default=5)
    # 指定每个客户端本地训练的周期数（即每个客户端在参与联邦学习时本地训练的次数）。
    parser.add_argument("--local_n_epochs", type=int, default=1)
    # 每个客户端在开始本地训练时会随机重初始化模型
    parser.add_argument("--random_reinit_local_model", default=None, type=str)
    # 指定本地代理项的系数，通常用于约束本地模型更新的幅度，防止本地训练过度偏离全局模型
    parser.add_argument("--local_prox_term", type=float, default=0)
    # 指定客户端进行本地训练的最小周期数。可以防止客户端进行过少的本地训练，影响训练质量
    parser.add_argument("--min_local_epochs", type=float, default=None)
    # 表示在每个训练周期结束后，数据将会重新洗牌（shuffle），以避免模型在训练过程中出现偏向性
    parser.add_argument("--reshuffle_per_epoch", default=False, type=str2bool)
    # 每次模型更新时使用的训练样本数,小批量训练
    parser.add_argument(
        "--batch_size",
        "-b",
        default=256,
        type=int,
        help="mini-batch size (default: 256)",
    )
    #
    parser.add_argument("--base_batch_size", default=None, type=int)
    # 指定联邦学习中的客户端数量
    parser.add_argument(
        "--n_clients",
        default=1,
        type=int,
        help="# of the clients for federated learning.",
    )
    # 指定每次通信回合中参与的客户端比例。值在 [0, 1] 之间，表示每次通信中有多少比例的客户端参与更新。
    parser.add_argument(
        "--participation_ratio",
        default=0.1,
        type=float,
        help="number of participated ratio per communication rounds",
    )
    # 如果指定了具体数量，则表示每回合参与的客户端数目，而非比例
    parser.add_argument("--n_participated", default=None, type=int)
    # 指定联邦学习的聚合策略（例如，FedAvg、FedProx 等）。该参数决定如何合并客户端的更新
    parser.add_argument("--fl_aggregate", default=None, type=str)
    # 非独立同分布数据的参数，通常用于模拟数据分布不均的情况
    parser.add_argument("--non_iid_alpha", default=0, type=float)
    # 启用训练加速选项
    parser.add_argument("--train_fast", type=str2bool, default=False)
    # 启用 MixUp 数据增强方法。MixUp 是一种通过线性插值生成新样本的技术，用于增强数据多样性。
    parser.add_argument("--use_mixup", default=False, type=str2bool)
    #  MixUp 数据增强方法中的 alpha 参数，通常控制插值的强度
    parser.add_argument("--mixup_alpha", default=1.0, type=float)
    # MixUp 方法将在非独立同分布的数据上应用
    parser.add_argument("--mixup_noniid", default=False, type=str2bool)


    # ----learning rate scheme
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="MultiStepLR",
        choices=["MultiStepLR", "ExponentialLR", "ReduceLROnPlateau"],
    )
    #   基础学习率、学习率调度器、里程碑、里程碑比例、衰减因子、耐心值、缩放、缩放初始值、缩放因子、预热、学习率预热的周期数、上限周期数
    parser.add_argument("--lr_milestones", type=str, default=None)
    parser.add_argument("--lr_milestone_ratios", type=str, default=None)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--lr_scaleup", type=str2bool, default=False)
    parser.add_argument("--lr_scaleup_init_lr", type=float, default=None)
    parser.add_argument("--lr_scaleup_factor", type=int, default=None)
    parser.add_argument("--lr_warmup", type=str2bool, default=False)
    parser.add_argument("--lr_warmup_epochs", type=int, default=None)
    parser.add_argument("--lr_warmup_epochs_upper_bound", type=int, default=150)
    # Adam 优化器中的参数
    parser.add_argument("--adam_beta_1", default=0.9, type=float)
    parser.add_argument("--adam_beta_2", default=0.999, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    # ---optimizer优化器：随机梯度下降
    parser.add_argument("--optimizer", type=str, default="sgd")
    # quantizer 是否启用本地模型压缩（如量化、剪枝等）
    parser.add_argument("--local_model_compression", type=str, default=None)
    # some SOTA training schemes, e.g., larc, label smoothing.训练方案
    # 启用 LARC（层次自适应学习率缩放）优化策略。这种方法根据每层的梯度大小自适应调整学习率，可以更好地处理深度神经网络中的训练问题。
    parser.add_argument("--use_larc", type=str2bool, default=False)
    # 设置 LARC 方法的信任系数，用于控制每个参数的更新幅度。值越大，优化器的学习率会更大。
    parser.add_argument("--larc_trust_coefficient", default=0.02, type=float)
    # 启用 LARC 的剪裁机制
    parser.add_argument("--larc_clip", default=True, type=str2bool)
    # 启用标签平滑（Label Smoothing）技术，用于缓解过拟合问题。标签平滑通过将目标标签分布从一个硬标签（如 0 或 1）转换为一个更软的分布来进行处理。
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    # 是否使用加权损失函数，适用于类别不平衡的任务，通过调整损失函数中不同类别的权重来平衡各类别的影响。
    parser.add_argument("--weighted_loss", default=None, type=str)
    # 设置加权损失函数中的加权系数
    parser.add_argument("--weighted_beta", default=0, type=float)
    parser.add_argument("--weighted_gamma", default=0, type=float)

    # momentum scheme动量策略
    # 动量优化器中的动量因子，控制前一个梯度的影响
    parser.add_argument("--momentum_factor", default=0.9, type=float)
    # 启用 Nesterov 动量优化算法。Nesterov 动量是在标准动量的基础上，使用前向梯度（预见）来进一步加速收敛。
    parser.add_argument("--use_nesterov", default=False, type=str2bool)

    # regularization正则化
    # 权重衰减，正则化的系数，也称为 L2 正则化。通过在损失函数中加入权重衰减项，可以避免模型权重过大，从而防止过拟合。
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="weight decay (default: 1e-4)"
    )
    # Dropout 层的丢弃率。Dropout 是一种常用的正则化方法，通过随机丢弃一部分神经元的激活值，来防止神经网络对训练数据的过拟合。
    parser.add_argument("--drop_rate", default=0.0, type=float)
    # 是否启用自蒸馏。自蒸馏是一种利用已经训练好的网络来指导新网络训练的方法。
    parser.add_argument("--self_distillation", default=0, type=float)
    # 控制自蒸馏中软标签的平滑程度。较高的温度会使得目标分布更加平滑。
    parser.add_argument("--self_distillation_temperature", default=1, type=float)

    # ---configuration for different models.不同模型的配置
    # 控制 DenseNet 模型中每一层的增长率
    parser.add_argument("--densenet_growth_rate", default=12, type=int)
    # 是否启用 DenseNet 的 Bottleneck 和 Compression 模式。该模式通过减少每个 DenseBlock 中的输出通道数来减少计算量
    parser.add_argument("--densenet_bc_mode", default=False, type=str2bool)
    # DenseNet 的压缩比例。用于压缩模型的大小，降低计算量，避免过拟合。
    parser.add_argument("--densenet_compression", default=0.5, type=float)
    # 控制 WideResNet 模型的宽度因子，即每一层的通道数相对增大的倍数。增大宽度因子可以提高模型的容量，但会增加计算量和内存占用。
    parser.add_argument("--wideresnet_widen_factor", default=4, type=int)


    # RNN 配置
    # 隐藏层单元数
    parser.add_argument("--rnn_n_hidden", default=200, type=int)
    # RNN 网络的层数。更多的层数可以提高模型的表达能力，但也会增加计算复杂度。
    parser.add_argument("--rnn_n_layers", default=2, type=int)
    # 反向传播的时间步长
    parser.add_argument("--rnn_bptt_len", default=35, type=int)
    # 梯度裁剪，防止梯度爆炸。指定一个最大值，如果梯度超过这个值，就进行裁剪。
    parser.add_argument("--rnn_clip", type=float, default=0.25)
    # 预训练的词嵌入
    parser.add_argument("--rnn_use_pretrained_emb", type=str2bool, default=True)
    # 共享权重：是否在 RNN 中共享输入和输出的权重
    parser.add_argument("--rnn_tie_weights", type=str2bool, default=True)
    # 权重归一化，帮助优化训练过程，特别是在 RNN 中，权重归一化有助于加速收敛。
    parser.add_argument("--rnn_weight_norm", type=str2bool, default=False)

    # Transformer 配置
    # Transformer 模型的层数。层数越多，模型的表达能力越强，但计算和内存消耗也越大。
    parser.add_argument("--transformer_n_layers", default=6, type=int)
    # 每一层中的注意力头数
    parser.add_argument("--transformer_n_head", default=8, type=int)
    # Transformer 模型的维度，即每一层的隐藏状态的维度。
    parser.add_argument("--transformer_dim_model", default=512, type=int)
    # 内部隐藏层维度
    parser.add_argument("--transformer_dim_inner_hidden", default=2048, type=int)
    # Transformer 中的预热步骤数，用于学习率逐步增加
    parser.add_argument("--transformer_n_warmup_steps", default=4000, type=int)

    # miscs
    # 相同随机种子,如果为 True，确保每个训练过程（每个进程）都使用相同的随机种子。这样可以确保实验的可重复性，避免由于不同进程使用不同随机种子而导致实验结果的差异。
    parser.add_argument("--same_seed_process", type=str2bool, default=True)
    # 手动设置随机种子，以确保每次运行的结果一致性。通常设置一个固定的整数作为种子，以便复现实验结果。
    parser.add_argument("--manual_seed", type=int, default=6, help="manual seed")
    # 是否进行模型评估
    parser.add_argument(
        "--evaluate",
        "-e",
        dest="evaluate",
        type=str2bool,
        default=False,
        help="evaluate model on validation set",
    )
    # 控制在训练过程中输出训练摘要的频率。例如，每训练 256 个样本输出一次摘要。
    parser.add_argument("--summary_freq", default=256, type=int)
    # 用于为每个实验生成一个时间戳。可以帮助用户标识不同的实验
    parser.add_argument("--timestamp", default=None, type=str)
    # 追踪时间：是否记录训练过程的时间。如果为 True，会记录训练开始、结束以及每个阶段所花费的时间。
    parser.add_argument("--track_time", default=False, type=str2bool)
    # 追踪详细时间
    parser.add_argument("--track_detailed_time", default=False, type=str2bool)
    # 是否在训练过程中显示时间追踪的结果
    parser.add_argument("--display_tracked_time", default=False, type=str2bool)

    # checkpoint模型检查点
    # 指定一个路径，用于从之前保存的检查点恢复模型的训练，而不是从头开始
    parser.add_argument("--resume", default=None, type=str)
    # 保存检查点的路径。训练过程中，模型会定期保存到该路径
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=TRAINING_DIRECTORY,
        type=str,
        help="path to save checkpoint (default: checkpoint)",
    )
    # 指定要保存的检查点的索引。可以用于保存特定的训练轮次的模型状态
    parser.add_argument("--checkpoint_index", type=str, default=None)
    # 是否保存所有模型，如果为 True，则每经过一定的训练轮次，就保存一个模型状态。
    parser.add_argument("--save_all_models", type=str2bool, default=False)
    # 指定一个列表，表示在某些特定的通信轮次（例如，在联邦学习中的每个轮次）保存模型
    parser.add_argument("--save_some_models", type=str, default=None, help="a list for comm_round to save")


    # device设备
    # Python 路径
    parser.add_argument(
        "--python_path", type=str, default="$HOME/conda/envs/pytorch-py3.6/bin/python"
    )
    # 定义一个设备列表，通常用于指定多个计算设备（如多个 GPU）或多个节点（在分布式训练中）
    parser.add_argument("--world", default=None, type=str, help="a list for devices.")
    parser.add_argument("--world_conf", default=None, type=str,
                        help="a list for the logic of world_conf follows a,b,c,d,e where: the block range from 'a' to 'b' with interval 'c' (and each integer will repeat for 'd' time); the block will be repeated for 'e' times.")
    # 是否使用 CUDA 进行 GPU 加速
    parser.add_argument("--on_cuda", type=str2bool, default=True)
    # 一个主机文件路径，用于定义分布式训练中的多个计算节点。该文件通常包含每个节点的网络地址或主机名
    parser.add_argument("--hostfile", type=str, default=None)
    #  MPI（消息传递接口）环境的路径。MPI 用于在多个计算节点或多个 GPU 之间进行通信，尤其是在分布式训练中
    parser.add_argument("--mpi_path", type=str, default="$HOME/.openmpi")
    #  MPI 的环境变量设置。它可以包含一些额外的配置，帮助优化分布式计算的通信方式。
    parser.add_argument("--mpi_env", type=str, default=None)
    """meta info."""
    # 实验的名称
    parser.add_argument("--experiment", type=str, default="debug")
    # 用于保存作业日志的路径
    parser.add_argument("--job_id", type=str, default="/tmp/jobrun_logs")
    # 指定训练脚本所在的路径
    parser.add_argument("--script_path", default="exp/", type=str)
    # 指定脚本中的类名
    parser.add_argument("--script_class_name", default=None, type=str)
    # 每个计算节点上运行的作业数量
    parser.add_argument("--num_jobs_per_node", default=1, type=int)
    """yaml"""
    # 训练配置的 YAML 文件路径
    parser.add_argument("--config_yaml", type=str, default="config.yaml")
    # parse conf.
    return parser


def parse_args_with_yaml(parser):
    '''使用YAML 配置文件来覆盖默认的配置参数，本项目应该是没有的'''
    args, unknown_args = parser.parse_known_args()
    # override default configurations with yaml file
    if args.config_yaml:
        with open(args.config_yaml, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args, unknown_args = parser.parse_known_args()

    key=None
    for item in unknown_args:
        if item.startswith('--'):
            if key:
                args.__setattr__(key, True)
            key = item[2:]
        else:
            value= item.replace('-', '_')
            if key:
                args.__setattr__(key, value)
                key = None
    return args


def save_config(args,yaml):
    '''用于将参数配置保存到yaml文件中'''
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    with open(yaml) as f:
        f.write(args_text)
    return args_text


def complete_missing_config(conf):
    '''检查给定的配置 conf 是否缺少某些关键字段，并根据需要为它们填充默认值'''
    if "port" not in conf or conf.port is None:
        conf.port = get_free_port()
    if not conf.n_participated:
        conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)
    conf.timestamp = str(int(time.time()))


# find free port for communication.
def get_free_port():
    """ Get free port for communication."""
    #  使用 closing 来确保 socket 在使用后正确关闭
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        # 将 socket 绑定到任意一个可用的端口上
        s.bind(('', 0))
        # 设置 socket 选项，允许端口复用
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 返回分配给套接字的端口号
        return str(s.getsockname()[1])


def validate_config(conf):
    if conf.backend == "nccl" and conf.workers + 1 <= torch.cuda.device_count():
        raise ValueError("The NCCL backend requires exclusive access to CUDA devices.")


def init_config(conf):
    '''用于初始化分布式训练环境和配置'''
    # define the graph for the computation.
    # 定义一个计算图（graph），通常用于分布式训练中每个节点（worker）之间的通信和计算依赖
    conf.graph = topology.define_graph_topology(
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )
    # dist.get_rank() 用来获取当前进程在分布式环境中的 rank（即当前节点的编号）
    conf.graph.rank = dist.get_rank()
    # 调用 set_random_seed 函数来初始化随机数生成器，确保实验可复现。
    set_random_seed(conf)
    # 调用 set_device 配置训练时使用的设备（
    set_device(conf)

    # init the model arch info.初始化模型架构信息
    conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
    conf.arch_info["worker"] = conf.arch_info["worker"].split(":")

    # parse the fl_aggregate scheme.解析联邦学习聚合配置
    conf._fl_aggregate = conf.fl_aggregate
    conf.fl_aggregate = (
        param_parser.dict_parser(conf.fl_aggregate)
        if conf.fl_aggregate is not None
        else conf.fl_aggregate
    )
    # 将聚合配置写入 conf 对象
    [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]

    # define checkpoint for logging (for federated learning server).初始化检查点
    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank))#

    # configure logger.在路徑创建一个日志记录
    conf.logger = logging.Logger(conf.checkpoint_dir)

    # display the arguments' info rank == 0 的节点（即主节点）上执行
    if conf.graph.rank == 0:
        logging.display_args(conf)

    # sync the processes.
    # 使用 dist.barrier() 同步分布式训练中的所有进程。该函数会阻塞所有进程，直到所有进程都到达此位置，从而保证不同进程间的同步。
    dist.barrier()


def set_random_seed(conf):
    '''通过设置随机种子来控制实验中的所有随机过程'''
    # init related to randomness on cpu.
    if not conf.same_seed_process:
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank
    # set seed to ensure experiment reproducibility.
    random.seed(conf.manual_seed)
    np.random.seed(conf.manual_seed)
    # 用于生成随机数的类。它允许用户设定一个 随机种子，从而确保每次生成的随机数序列都是相同的
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)
    torch.cuda.manual_seed(conf.manual_seed)
    try:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception as e:
        traceback.print_exc()


def set_device(conf):
    '''为分布式训练环境中的每个进程选择合适的设备（通常是 GPU）。它根据配置 conf.on_cuda 来决定是否使用 CUDA（即 GPU），并通过一些逻辑来选择具体的设备。
    该函数在分布式训练中非常有用，确保每个进程根据其 rank 和设备的内存使用情况来选择合适的 GPU'''
    if conf.on_cuda:
        # set cuda to ensure experiment reproducibility.
        if dist.get_backend() == "nccl":
            torch.cuda.set_device(torch.device("cuda:" + str(dist.get_rank())))
        else:
            pynvml.nvmlInit() #pynvml.nvmlInit() 初始化 NVML（NVIDIA Management Library），该库可以提供 GPU 设备的信息，包括每个 GPU 的可用内存
            device_count = pynvml.nvmlDeviceGetCount() #获取可用的 GPU 数量
            available_memory = [] #用来存储每个 GPU 的空闲内存
            # 获取每个 GPU 的可用内存
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i) #获取第 i 个 GPU 的句柄
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle) #获取该 GPU 的内存信息。
                available_memory.append(mem_info.free) #mem_info.free表示该 GPU 的空闲内存，空闲内存被添加到 available_memory 列表中
            available_memory = np.array(available_memory)
            # 计算 GPU 内存分配比例
            available_memory_patition = (available_memory / available_memory.sum()).cumsum()
            # 计算进程应选择的设备位置
            device_position = (dist.get_rank()) / dist.get_world_size() #表示当前进程在所有进程中所占的比例
            for i, patition in enumerate(available_memory_patition):
                # 如果 device_position 小于等于某个 GPU 的内存占比，就选择这个 GPU。
                if device_position <= patition:
                    break
            torch.cuda.set_device(torch.device("cuda:" + str(i))) #设置当前进程使用的 GPU
        # torch.cuda.set_device(torch.device("cuda:" + str(conf.graph.rank % torch.cuda.device_count())))
    else:
        torch.cuda.set_device(torch.device("cpu"))

def set_environ():
    # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # 如果运行环境（例如你的服务器或本地计算机）所在的网络环境无法直接访问外网（例如因为防火墙或网络限制），配置代理可以绕过这些限制，使程序可以访问外部服务。
    # os.environ['all_proxy'] = 'http://202.114.7.49:7890'
    os.environ['WANDB_MODE'] = 'online'

def experiment_paramenter(conf):
    if conf.data == 'music':
        conf.neighbor_sample_size = 8
        conf.dim = 16
        conf.n_iter = 1 # 迭代次数
        conf.weight_decay = 1e-5
        conf.lr = 5e-5
        conf.batch_size = 32
    elif conf.data == 'book':
        conf.neighbor_sample_size = 8
        conf.dim = 64
        conf.n_iter = 1
        conf.weight_decay = 2e-5
        conf.lr = 2e-4
        conf.batch_size = 32
    elif conf.data == 'movie':
        conf.neighbor_sample_size = 4
        conf.dim = 32
        conf.n_iter = 2
        conf.weight_decay = 1e-7
        conf.lr = 2e-2
        conf.batch_size = 32

# add debug environment
def debug_parameter(conf):
    # debug
    debug=False

    # conf.data = 'book'
    if debug==True:
        os.environ['WANDB_MODE'] = 'offline'
        conf.n_participated = 4
        conf.workers = 4
        conf.validation_interval = 1
        conf.topk_eval_interval = 1
    else:
        conf.n_participated = 32
        conf.workers = 32
        conf.validation_interval = 10
        conf.topk_eval_interval =30
    conf.train_fast = True
    conf.backend = "gloo"

    conf.n_comm_rounds = 2000*32
    conf.aggregator = "sum"
    conf.same_arch=True
    conf.experiment=f'fedKgcn_dataset_{conf.data}_np_{conf.n_participated}_nc_{conf.n_comm_rounds}'
    conf.k_list= [20, 50, 100]
    conf.local_batch_size = None
    return conf
