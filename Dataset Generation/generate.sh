gpu=6
for lm_name in gpt2 gpt2-medium
do
    for dataset in climate graphika avax
        do
        cmd="python train.py --lm_name $lm_name --dataset $dataset --model_batch_size 32 --mode generate --num_samples 10000 --gpu $gpu"
        echo $cmd
        $cmd
        # let gpu=gpu+1
        done
done



# gpu=0
# for lm_name in gpt2-large EleutherAI/gpt-neo-1.3B
# do
#     for dataset in climate graphika avax
#         do
#         cmd="python train.py --lm_name $lm_name --dataset $dataset --model_batch_size 32 --mode generate --num_samples 10000 --gpu $gpu"
#         echo $cmd
#         $cmd &
#         let gpu=gpu+1
#         done
# done