protein=uricase_search
node_s_type=foldseek
python src/data/build_graph.py \
    --pdb_dir data/$protein/pdb \
    --node_level residue \
    --node_s_type $node_s_type \
    --max_distance 10 \
    --graph_out_dir data/$protein/graph_$node_s_type \
    --error_file data/$protein/graph_"$node_s_type"_error.csv

protein=cath_v43_s20
node_s_type=foldseek_ss8
python src/data/build_graph.py \
    --pdb_dir data/$protein/dompdb \
    --node_level residue \
    --node_s_type $node_s_type \
    --foldseek_fasta_file data/$protein/foldseek/queryDB_ss.fasta \
    --foldseek_fasta_multi_chain \
    --max_distance 10 \
    --graph_out_dir data/$protein/graph_$node_s_type \
    --error_file data/$protein/graph_"$node_s_type"_error.csv

protein=cath_v43_s20
node_s_type=ss8
python src/data/build_graph.py \
    --pdb_dir data/$protein/dompdb \
    --node_level residue \
    --node_s_type $node_s_type \
    --max_distance 10 \
    --graph_out_dir data/$protein/graph_$node_s_type \
    --error_file data/$protein/graph_"$node_s_type"_error.csv

protein=cath_v43_s20
node_s_type=ss3
python src/data/build_graph.py \
    --pdb_dir data/$protein/dompdb \
    --node_level residue \
    --node_s_type $node_s_type \
    --max_distance 10 \
    --graph_out_dir data/$protein/graph_$node_s_type \
    --error_file data/$protein/graph_"$node_s_type"_error.csv

protein=cath_v43_s20
node_s_type=aa
python src/data/build_graph.py \
    --pdb_dir data/$protein/dompdb \
    --node_level residue \
    --node_s_type $node_s_type \
    --max_distance 10 \
    --graph_out_dir data/$protein/graph_$node_s_type \
    --error_file data/$protein/graph_"$node_s_type"_error.csv

protein=cath_v43_s20
node_s_type=aa_foldseek
python src/data/build_graph.py \
    --pdb_dir data/$protein/dompdb \
    --node_level residue \
    --node_s_type $node_s_type \
    --foldseek_fasta_file data/$protein/foldseek/queryDB_ss.fasta \
    --foldseek_fasta_multi_chain \
    --max_distance 10 \
    --graph_out_dir data/$protein/graph_$node_s_type \
    --error_file data/$protein/graph_"$node_s_type"_error.csv

protein=cath_v43_s40
node_s_type=aa_ss8
python src/data/build_graph.py \
    --pdb_dir data/$protein/dompdb \
    --node_level residue \
    --node_s_type $node_s_type \
    --max_distance 10 \
    --graph_out_dir data/$protein/graph_$node_s_type \
    --error_file data/$protein/graph_"$node_s_type"_error.csv

protein=cath_v43_s20
node_s_type=aa_foldseek_ss8
python src/data/build_graph.py \
    --pdb_dir data/$protein/dompdb \
    --node_level residue \
    --node_s_type $node_s_type \
    --foldseek_fasta_file data/$protein/foldseek/queryDB_ss.fasta \
    --foldseek_fasta_multi_chain \
    --max_distance 10 \
    --graph_out_dir data/$protein/graph_$node_s_type \
    --error_file data/$protein/graph_"$node_s_type"_error.csv

protein=3DAC
node_s_type=foldseek
python src/data/build_graph.py \
    --pdb_dir data/$protein/pdb \
    --node_level residue \
    --node_s_type $node_s_type \
    --foldseek_fasta_file data/$protein/foldseek/queryDB_ss.fasta \
    --max_distance 10 \
    --graph_out_dir data/$protein/graph_$node_s_type \
    --error_file data/$protein/graph_"$node_s_type"_error.csv

protein=hyt_ECAI2024
node_s_type=ss3
python src/data/build_graph.py \
    --pdb_dir data/$protein/pdb \
    --node_level secondary_structure \
    --node_s_type $node_s_type \
    --foldseek_fasta_file data/$protein/foldseek/queryDB_ss.fasta \
    --max_distance 20 \
    --graph_out_dir data/$protein/graph_$node_s_type \
    --error_file data/$protein/graph_"$node_s_type"_error.csv
