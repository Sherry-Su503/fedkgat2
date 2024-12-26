# FedRKG: A Privacy-preserving Federated Recommendation Framework via Knowledge Graph Enhancement

This repository is the official implementation of the paper *FedRKG: A Privacy-preserving Federated Recommendation Framework
via Knowledge Graph Enhancement*.

**Abstract:**
*Federated Learning* (FL) has emerged as a promising approach for preserving data privacy in recommendation
systems by training models locally. Recently, *Graph Neural Networks* (GNN) have gained popularity in
recommendation tasks due to their ability to capture high-order interactions between users and items. However, privacy
concerns prevent the global sharing of the entire user-item graph. To address this limitation, some methods create
pseudo-interacted items or users in the graph to compensate for missing information for each client. Unfortunately,
these methods introduce random noise and raise privacy concerns.
In this paper, we propose FedRKG, a novel federated recommendation system, where a global *knowledge graph* (KG) is
constructed and maintained on the server using publicly available item information, enabling higher-order user-item
interactions.
On the client side, a relation-aware GNN model leverages diverse KG relationships.
To protect local interaction items and obscure gradients, we employ pseudo-labeling and *Local Differential
Privacy* (LDP).
Extensive experiments conducted on three real-world datasets demonstrate the competitive performance of our approach
compared to centralized algorithms while ensuring privacy preservation. Moreover, FedRKG achieves an average accuracy
improvement of 4% compared to existing federated learning baselines.
## Requirements

Our implementations heavily rely on `Docker` and the detailed environment setup refers to `Dockerfile` under
the `../environments` folder.

By running command `docker-compose build` under the folder `environments`, you can build our main docker
image `pytorch-mpi`.

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
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

music   --n-client 1065
python run.py --arch kgcn --complex_arch master=kgcn_kg,worker=kgcn_aggregate --experiment serial --data music --pin_memory True --batch_size 32 --num_workers 1 --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0 --n_clients 1065 --participation_ratio 1 --n_comm_rounds 2000 --local_n_epochs 1 --world_conf 0,0,1,1,100 --on_cuda True --fl_aggregate scheme=federated_average --optimizer adam --lr 5e-4 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 --lr_scheduler MultiStepLR --lr_decay 0.1 --weight_decay 1e-4 --use_nesterov False --momentum_factor 0 --track_time False --display_tracked_time False --hostfile hostfile --manual_seed 7 --pn_normalize True --same_seed_process False --python_path /root/miniconda3/bin/python


book  --n-client 1113
python run.py --arch kgcn --complex_arch master=kgcn_kg,worker=kgcn_aggregate --experiment serial --data book --pin_memory True --batch_size 32 --num_workers 1 --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0 --n_clients 1113 --participation_ratio 1 --n_comm_rounds 2000 --local_n_epochs 1 --world_conf 0,0,1,1,100 --on_cuda True --fl_aggregate scheme=federated_average --optimizer adam --lr 5e-4 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 --lr_scheduler MultiStepLR --lr_decay 0.1 --weight_decay 1e-4 --use_nesterov False --momentum_factor 0 --track_time False --display_tracked_time False --hostfile hostfile --manual_seed 7 --pn_normalize True --same_seed_process False --python_path /root/miniconda3/bin/python

报错现存不够，需要减小批次

movie  --n-client 137728
python run.py --arch kgcn --complex_arch master=kgcn_kg,worker=kgcn_aggregate --experiment serial --data movie --pin_memory True --batch_size 32 --num_workers 1 --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0 --n_clients 137728 --participation_ratio 1 --n_comm_rounds 2000 --local_n_epochs 1 --world_conf 0,0,1,1,100 --on_cuda True --fl_aggregate scheme=federated_average --optimizer adam --lr 5e-4 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 --lr_scheduler MultiStepLR --lr_decay 0.1 --weight_decay 1e-4 --use_nesterov False --momentum_factor 0 --track_time False --display_tracked_time False --hostfile hostfile --manual_seed 7 --pn_normalize True --same_seed_process False --python_path /root/miniconda3/bin/python





梯度大小：
model_grad
tensor(-0.0022)
tensor(0.0264)
tensor(-0.0063)
tensor(0.0161)

embeddings_grad
tensor(-8.2561e-06)
tensor(-3.9678e-06)


model_grad
tensor(-3.9678e-06)
tensor(0.0012)
tensor(0.0105)
tensor(0.0068)
tensor(0.0057)
tensor(5.5765e-05)
tensor(1.0834e-05)
embeddings_grad
tensor(5.5765e-05)
tensor(1.0834e-05)