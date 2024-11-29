import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import traceback
from distutils import dist
import torch.multiprocessing as mp
from parameters import get_args, init_config
from pcode.master import Master
from pcode.utils.auto_distributed import *
from pcode.worker import Worker


# -*- coding: utf-8 -*-


def run(conf):
    # federated learning  function
    # conf.graph.rank 是进程的排名？
    # 主节点，和工作节点
    process = Master(conf) if conf.graph.rank == 0 else Worker(conf)
    process.run()


def init_process(rank, size, fn, conf):
    '''用于初始化分布式训练的环境，设置进程组，加载配置并启动联邦学习（或分布式训练）的任务。'''
    # init the distributed world.
    try:
        # 设置 MASTER_ADDR 和 MASTER_PORT 环境变量，指定主节点的地址和端口
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = conf.port
        # 初始化分布式进程组，使用指定的后端（如 gloo 或 nccl）
        dist.init_process_group(conf.backend, rank=rank, world_size=size)
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False
    try:
        # 忽略 UserWarning 警告，避免干扰输出
        warnings.filterwarnings("ignore", category=UserWarning)

        # init the config.init_config(conf)
        init_config(conf)

        # start federated learning.
        fn(conf)

    except Exception as e:
        print(f"Caught exception in rank {rank}")
        traceback.print_exc()
        raise e


def is_mpi_enabled():
    return 'MPI_COMM_WORLD_SIZE' in os.environ


def set_working_directory():
    current_file = os.path.abspath(__file__)
    directory = os.path.dirname(current_file)
    os.chdir(directory)


def run_mpi():
    if is_mpi_enabled():
        init_process(0, 0, run, conf)
    else:
        os.environ['MPI_COMM_WORLD_SIZE'] = size.__str__()
        args_str = ' '.join(sys.argv[1:])
        python_prefix = sys.prefix
        os.system(
            f'$HOME/.openmpi/bin/mpirun -n {size} --mca orte_base_help_aggregate 0 --mca btl_tcp_if_exclude docker0,lo --hostfile {conf.hostfile} {python_prefix}/bin/python run_gloo.py ' + args_str)


def run_gloo():
    '''启动多个进程'''
    processes = [] #用于存储所有进程对象
    for rank in range(size): # 迭代根据大小（`size`）来启动进程
        # 每个进程都会执行 init_process 函数，init_process 是你自己定义的函数，它会在每个进程中被调用。args 是传递给 init_process 函数的参数
        # rank: 进程的标识符，size: 总进程数
        # run: 分布式训练的核心任务，通常是训练模型的代码
        # conf: 配置对象，包含了所有的训练参数和设置。
        p = mp.Process(target=init_process, args=(rank, size, run, conf)) # 为每个进程创建一个 Process 实例
        p.start() # 启动进程
        processes.append(p)
    for p in processes: # 等待所有进程执行完毕
        p.join() # 等待每个进程完成

if __name__ == "__main__":
    # get config.

    # 获取超参数
    conf = get_args()

    # 设置路径
    set_working_directory()

    # Create process for each worker and master.
    # number of data loading workers (default: 4)
    size = conf.workers + 1

    # torch.multiprocessing.set_start_method("spawn")是PyTorch中的一个函数，用于设置多进程的启动方法。
    # "spawn"是一种启动方法，它创建一个新的Python解释器进程来执行子任务。
    # 这种方法适用于大多数情况，因为它可以避免全局解释器锁（GIL）的限制，允许多个进程同时运行。
    mp.set_start_method("spawn")

    # MPI（Message Passing Interface）是一种并行计算的标准接口，用于在分布式系统中进行进程间通信和同步
    if conf.backend == 'mpi':
        run_mpi()

    elif conf.backend in ['gloo','nccl']:
        run_gloo()
