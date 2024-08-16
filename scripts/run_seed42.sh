python mafed/train.py \
    --config config/train-vqa-base-cl-local-vlpythia.json \
    --exp question_types \
    --cl_method naive \
    --seed 42 \
    --tasks action count subcat scene color \
    --model_name storage/models/vl-pythia-eva-1b \
    --run_group question_types_naive_vl-pythia-eva-1b \
    --run_name vl-pythia-eva-1b_seed42 \
    --output_dir storage/question_types/seed42/vl-pythia-eva-1b_bsz128_lr5e-5_naive \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --accumulate_grad_batches 4 \
    --n_workers 4

# EWC
python mafed/train.py \
    --config config/train-vqa-base-cl-local-vlpythia.json \
    --exp question_types \
    --cl_method ewc \
    --reg_lambda 10000 \
    --seed 42 \
    --tasks action count subcat scene color \
    --model_name storage/models/vl-pythia-eva-1b \
    --run_group question_types_ewc_vl-pythia-eva-1b \
    --run_name vl-pythia-eva-1b_seed42 \
    --output_dir storage/question_types/seed42/vl-pythia-eva-1b_bsz128_lr5e-5_ewc \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --accumulate_grad_batches 4 \
    --n_workers 4

# Experience Replay
python mafed/train.py \
    --config config/train-vqa-base-cl-local-vlpythia.json \
    --exp question_types \
    --cl_method replay \
    --replay_interval 4 \
    --seed 42 \
    --tasks action count subcat scene color \
    --model_name storage/models/vl-pythia-eva-1b \
    --run_group question_types_replay_vl-pythia-eva-1b \
    --run_name vl-pythia-eva-1b_seed42 \
    --output_dir storage/question_types/seed42/vl-pythia-eva-1b_bsz128_lr5e-5_replay \
    --learning_rate 5e-5 \
    --batch_size 32 \
    --accumulate_grad_batches 4 \
    --n_workers 4

# Feature Distillation + Experience Replay
python mafed/train.py \
    --config config/train-vqa-base-cl-local-vlpythia.json \
    --exp question_types \
    --cl_method featdistill \
    --distillation_modality_weighing_strategy equal \
    --distillation_coeff 1 \
    --distillation_layer_weighing_strategy discounted \
    --distillation_layer_discount 0.5 \
    --replay_interval 4 \
    --replay_coeff 1 \
    --seed 42 \
    --tasks action count subcat scene color \
    --model_name storage/models/vl-pythia-eva-1b \
    --run_group question_types_featdistill_vl-pythia-eva-1b \
    --run_name vl-pythia-eva-1b_seed42 \
    --output_dir storage/question_types/seed42/vl-pythia-eva-1b_bsz128_lr5e-5_featdistill \
    --learning_rate 5e-5 \
    --batch_size 16 \
    --accumulate_grad_batches 4 \
    --n_workers 4


# Modality Aware Feature Distillation - Balanced + Experience Replay
python mafed/train.py \
    --config config/train-vqa-base-cl-local-vlpythia.json \
    --exp question_types \
    --cl_method featdistill \
    --distillation_modality_weighing_strategy balanced \
    --distillation_coeff 1 \
    --distillation_layer_weighing_strategy discounted \
    --distillation_layer_discount 0.5 \
    --replay_interval 4 \
    --replay_coeff 1 \
    --seed 42 \
    --tasks action count subcat scene color \
    --model_name storage/models/vl-pythia-eva-1b \
    --run_group question_types_mafedB_vl-pythia-eva-1b \
    --run_name vl-pythia-eva-1b_seed42 \
    --output_dir storage/question_types/seed42/vl-pythia-eva-1b_bsz128_lr5e-5_mafedB \
    --learning_rate 5e-5 \
    --batch_size 16 \
    --accumulate_grad_batches 4 \
    --n_workers 4

# Modality Aware Feature Distillation - Adaptive + Experience Replay
python mafed/train.py \
    --config config/train-vqa-base-cl-local-vlpythia.json \
    --exp question_types \
    --cl_method featdistill \
    --distillation_modality_weighing_strategy adaptive \
    --distillation_coeff 1 \
    --distillation_layer_weighing_strategy discounted \
    --distillation_layer_discount 0.5 \
    --replay_interval 4 \
    --replay_coeff 1 \
    --seed 42 \
    --tasks action count subcat scene color \
    --model_name storage/models/vl-pythia-eva-1b \
    --run_group question_types_mafedA_vl-pythia-eva-1b \
    --run_name vl-pythia-eva-1b_seed42 \
    --output_dir storage/question_types/seed42/vl-pythia-eva-1b_bsz128_lr5e-5_mafedA \
    --learning_rate 5e-5 \
    --batch_size 16 \
    --accumulate_grad_batches 4 \
    --n_workers 4