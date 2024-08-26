#!/bin/bash

dataset="rocket"          # 数据集名称 (book/taobao/rocket)
model_type="ComiRec-SA"   # 模型类型 (DNN/GRU4REC/MIND/ComiRec-DR/ComiRec-SA/UMI)
 
# for num_interest in 4 8; do
#     for batch_size in 128 512; do
#         for learning_rate in 0.01 0.001 0.005; do
#             echo "Training model with num_interest: $num_interest, batch_size:$batch_size, and learning_rate: $learning_rate"

#             python3 src/train.py \
#                 --dataset $dataset \
#                 --num_interest $num_interest \
#                 --batch_size $batch_size \
#                 --model_type $model_type \
#                 --learning_rate $learning_rate
#         done
#     done
# done

python src/train.py --dataset $dataset --model_type $model_type
python src/train.py --dataset $dataset --model_type $model_type
#shutdown -h now