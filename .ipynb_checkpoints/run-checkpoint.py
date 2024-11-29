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
    process = Master(conf) if conf.graph.rank == 0 else Worker(conf)
    process.run()


def init_process(rank, size, fn, conf):
    # init the distributed world.
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = conf.port
        dist.init_process_group(conf.backend, rank=rank, world_size=size)
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False
    try:
        warnings.filterwarnings("ignore", category=UserWarning)

        # init the config.
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
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, conf))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

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
