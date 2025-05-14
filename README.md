# branch:fedkgcn_mv
KG-FedCGNN模型，适用于movieLens-20M数据集训练
```bash
clone https://github.com/Sherry-Su503/fedkgat2.git
git branch kg-fedcgnn-mv
git pull origin fedkgat_mv
```

# KG-FedCGNN

原始代码参考了 paper *FedRKG: A Privacy-preserving Federated Recommendation Framework
via Knowledge Graph Enhancement*.

**Abstract:**

## 远程仓库分支说明
**fedgnn_u**: 跨用户隐私保护推荐模型，对应论文FedCGNN模型，适用于Last.fm和Book-Crossing数据集训练（小数据集，联邦通信时直接传输模型.

**fedgnn_u_mv**: 跨用户隐私保护推荐模型_movieLens-20M训练版，对应论文FedCGNN模型，适用于movieLens-20M数据集训练（大数据集，联邦通信时传输模型参数梯度，客户端本地自己初始化模型.

**fedkgcn**: 本地知识图谱扩展的可解释推荐模型，对应论文KG-FedCGNN模型，适用于Last.fm和Book-Crossing数据集训练（小数据集，联邦通信时直接传输模型）.

**Fekgcn_mv**: 本地知识图谱扩展的可解释推荐模型_movieLens-20M训练版，对应论文KG-FedCGNN模型，适用于movieLens-20M数据集训练（大数据集，联邦通信时传输模型参数梯度，客户端本地自己初始化模型）.

## Requirements


## Training and Evaluation

To train and evaluate the model(s) in the paper, run the following commands.
```bash
python run.py --arch kgcn --complex_arch master=kgcn_kg,worker=kgcn_aggregate --experiment serial --data music --pin_memory True --batch_size 32 --num_workers 1 --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0 --n_clients 1872 --participation_ratio 1 --n_comm_rounds 2000 --local_n_epochs 1 --world_conf 0,0,1,1,100 --on_cuda True --fl_aggregate scheme=federated_average --optimizer adam --lr 5e-4 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 --lr_scheduler MultiStepLR --lr_decay 0.1 --weight_decay 1e-4 --use_nesterov False --momentum_factor 0 --track_time False --display_tracked_time False --hostfile hostfile --manual_seed 7 --pn_normalize True --same_seed_process False --python_path /root/miniconda3/bin/python
```

```
git config --global user.email 2745043515@qq.com
git config --global user.name Sherry-Su503
  ```
Sherry-Su503
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

## dataset
music   --n-client 1065

```bash
python run.py --arch kgcn --complex_arch master=kgcn_kg,worker=kgcn_aggregate --experiment serial --data music --pin_memory True --batch_size 32 --num_workers 1 --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0 --n_clients 1065 --participation_ratio 1 --n_comm_rounds 2000 --local_n_epochs 1 --world_conf 0,0,1,1,100 --on_cuda True --fl_aggregate scheme=federated_average --optimizer adam --lr 5e-4 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 --lr_scheduler MultiStepLR --lr_decay 0.1 --weight_decay 1e-4 --use_nesterov False --momentum_factor 0 --track_time False --display_tracked_time False --hostfile hostfile --manual_seed 7 --pn_normalize True --same_seed_process False --python_path /root/miniconda3/bin/python
```


book  --n-client 1113
```bash
python run.py --arch kgcn --complex_arch master=kgcn_kg,worker=kgcn_aggregate --experiment serial --data book --pin_memory True --batch_size 32 --num_workers 1 --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0 --n_clients 1113 --participation_ratio 1 --n_comm_rounds 2000 --local_n_epochs 1 --world_conf 0,0,1,1,100 --on_cuda True --fl_aggregate scheme=federated_average --optimizer adam --lr 5e-4 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 --lr_scheduler MultiStepLR --lr_decay 0.1 --weight_decay 1e-4 --use_nesterov False --momentum_factor 0 --track_time False --display_tracked_time False --hostfile hostfile --manual_seed 7 --pn_normalize True --same_seed_process False --python_path /root/miniconda3/bin/python
```

movie  --n-client 137728
```bash
python run.py --arch kgcn --complex_arch master=kgcn_kg,worker=kgcn_aggregate --experiment serial --data movie --pin_memory True --batch_size 32 --num_workers 1 --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0 --n_clients 137728 --participation_ratio 1 --n_comm_rounds 2000 --local_n_epochs 1 --world_conf 0,0,1,1,100 --on_cuda True --fl_aggregate scheme=federated_average --optimizer adam --lr 5e-4 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 --lr_scheduler MultiStepLR --lr_decay 0.1 --weight_decay 1e-4 --use_nesterov False --momentum_factor 0 --track_time False --display_tracked_time False --hostfile hostfile --manual_seed 7 --pn_normalize True --same_seed_process False --python_path /root/miniconda3/bin/python
```


