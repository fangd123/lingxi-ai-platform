export OUTPUT_DIR=speaker_model-T3
export BATCH_SIZE=128
export NUM_EPOCHS=30
export SAVE_STEPS=1750
export SEED=42
export MAX_LENGTH=128
export BERT_MODEL_TEACHER=../../../output_dir_ner
python run_ner_distill.py \
--data_dir ../../../data \
--model_type bert \
--model_name_or_path $BERT_MODEL_TEACHER \
--model_name_or_path_student /nfs/protech/模型库/预训练模型/script_novel_pretrain_bert_rbt3/ \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--num_hidden_layers 3 \
--save_steps $SAVE_STEPS \
--learning_rate 1e-4 \
--warmup_steps 0.1 \
--seed $SEED \
--do_train \
--do_distill \
--do_eval \
--do_predict \
--labels "../../../data/labels.txt"
