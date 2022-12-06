python train.py --lm_name gpt2 --dataset avax --gpu 5 --model_batch_size 32
python train.py --lm_name gpt2 --dataset climate --gpu 5 --model_batch_size 32
python train.py --lm_name gpt2 --dataset graphika --gpu 5 --model_batch_size 32



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