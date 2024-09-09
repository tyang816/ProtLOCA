qsub -v index=0 -N AF19M_s_0 script/inference/get_struct_seq_afdb.pbs

for chunk_id in {0..188}
do
    qsub -v index=$chunk_id -N AF19M_s_$chunk_id script/inference/get_struct_seq_afdb.pbs
done
