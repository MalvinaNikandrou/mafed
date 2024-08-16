python  mafed/train.py \
    --config config/train-vqa-base-cl-local-vlpythia.json \
    --exp question_types \
    --cl_method naive \
    --seed 42 \
    --tasks action count subcategory scene color \
    --model_name storage/models/vl-pythia-eva-1b \
    --run_group question_types_naive_vl-pythia-eva-1b \
    --run_name vl-pythia-eva-1b_question_types_seed42 \
    --output_dir storage/question_types/seed42/vl-pythia-eva-1b_bsz128_lr5e-5_naive \
    --learning_rate 5e-5 \
    --batch_size 64 \
    --accumulate_grad_batches 2 \
    --n_workers 4

python  mafed/train.py \
    --config config/train-vqa-base-cl-local-vlpythia.json \
    --exp question_types \
    --cl_method naive \
    --seed 191 \
    --tasks color subcategory action count scene \
    --model_name storage/models/vl-pythia-eva-1b \
    --run_group question_types_naive_vl-pythia-eva-1b \
    --run_name vl-pythia-eva-1b_question_types_seed191 \
    --output_dir storage/question_types/seed191/vl-pythia-eva-1b_bsz128_lr5e-5_naive \
    --learning_rate 5e-5 \
    --batch_size 64 \
    --accumulate_grad_batches 2 \
    --n_workers 4

python  mafed/train.py \
    --config config/train-vqa-base-cl-local-vlpythia.json \
    --exp question_types \
    --cl_method naive \
    --seed 23 \
    --tasks scene count action color subcategory \
    --model_name storage/models/vl-pythia-eva-1b \
    --run_group question_types_naive_vl-pythia-eva-1b \
    --run_name vl-pythia-eva-1b_question_types_seed23 \
    --output_dir storage/question_types/seed23/vl-pythia-eva-1b_bsz128_lr5e-5_naive \
    --learning_rate 5e-5 \
    --batch_size 64 \
    --accumulate_grad_batches 2 \
    --n_workers 4