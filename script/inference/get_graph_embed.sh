# get_embed from subgraph dir
protein=cath_v43_s20
pdb_dir=/public/home/tanyang/workspace/SubProMatch/data/$protein/dompdb
CUDA_VISIBLE_DEVICES=0 python get_embed.py \
    --model_path ckpt/foldseek/20240119/foldseek_0.5m_50e.pt \
    --max_batch_nodes 3000 \
    --pdb_dir $pdb_dir \
    --num_processes 6 \
    --pooling_method mean \
    --out_file result/$protein/foldseek_0.5m_50e_sum_test.pt

protein=uricase_search
pdb_dir=/home/tanyang/workspace/SubProMatch/data/$protein/TIM_align_refined
CUDA_VISIBLE_DEVICES=1 python get_embed.py \
    --model_path ckpt/foldseek/20240119/foldseek_0.5m_50e.pt \
    --max_batch_nodes 10000 \
    --pdb_dir $pdb_dir \
    --num_processes 6 \
    --cache_graph \
    --out_file result/$protein/TIM_align_refined.pt

# get_embed from pdb dir
data_dir=ipr_motif_s20
pdb_dir=motif_pdb
CUDA_VISIBLE_DEVICES=0 python get_embed.py \
    --model_path ckpt/foldseek/20240119/foldseek_0.5m_50e.pt \
    --max_batch_nodes 3000 \
    --pdb_dir /public/home/tanyang/workspace/SubProMatch/data/$data_dir/$pdb_dir \
    --cache_graph \
    --rm_cache \
    --num_processes 12 \
    --pooling_method mean \
    --out_file result/$data_dir/$pdb_dir.3.pt