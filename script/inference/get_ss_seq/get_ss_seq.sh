
python get_ss_seq.py \
    --pdb_dir data/proteingym_pdb \
    --num_workers 6 \
    --out_file result/ss_seq/proteingym_dssp.json


pdb_dir=/home/tanyang/workspace/benchmark/ProteinCollector/data/AF19M/pdb
pdb_index_file_dir=/home/tanyang/workspace/benchmark/ProteinCollector/data/AF19M/chunks
CUDA_VISIBLE_DEVICES=0 python get_ss_seq.py \
    --pdb_dir $pdb_dir \
    --pdb_index_file $pdb_index_file_dir/uniprot_0.txt \
    --pdb_index_level 2 \
    --num_workers 6 \
    --out_file result/ss_seq/uniprot_0.json