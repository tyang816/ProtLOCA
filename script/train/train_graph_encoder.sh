node_type=foldseek
wandb_project=ProtLOCA
max_train_epochs=50
train_mask_node_ratio=0.5
num_layers=6

CUDA_VISIBLE_DEVICES=0 python train_graph_encoder.py \
    --node_s_type $node_type \
    --train_graph_dir data/cath_v43_s40/graph_$node_type \
    --train_mask_node_ratio $train_mask_node_ratio \
    --train_permute_node_ratio 0.5 \
    --valid_graph_num 200 \
    --test_graph_dir data/cath_v43_s20/graph_$node_type \
    --test_graph_pair_json benchmark/cath_v43_s20/domain_pair_dict_1.json \
    --test_mask_node \
    --max_train_epochs $max_train_epochs \
    --max_batch_nodes 10000 \
    --num_layers $num_layers \
    --lr 1e-4 \
    --num_workers 12 \
    --patience 5 \
    --ckpt_root ckpt/$node_type \
    --wandb \
    --wandb_project $wandb_project \
    --wandb_run_name "$node_type"_"$train_mask_node_ratio"m_"$num_layers"l_"$max_train_epochs"e