qsub -v index=0 -N AF19M_s_0 script/inference/get_ss_seq/get_ss_seq.pbs

for chunk_id in {0..188}
do
    qsub -v index=$chunk_id -N AF19M_s_$chunk_id script/inference/get_ss_seq/get_ss_seq.pbs
done