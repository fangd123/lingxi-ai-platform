#set hyperparameters
BERT_DIR=../../../output_dir
OUTPUT_ROOT_DIR=output_root_dir
DATA_ROOT_DIR=../../../data
trained_teacher_model=../../../output_dir/pytorch_model.bin
student_init_model=/nfs/protech/模型库/预训练模型/script_novel_pretrain_bert_rbt3/pytorch_model.bin
#STUDENT_CONF_DIR=../student_config/bert_base_cased_config
STUDENT_CONF_DIR=/nfs/protech/模型库/预训练模型/script_novel_pretrain_bert_rbt3/

accu=1
ep=60
lr=10
temperature=8
batch_size=128
length=128
sopt1=30 # The final learning rate is 1/sopt1 of the initial learning rate
torch_seed=42

taskname='mnli'
NAME=${taskname}_t${temperature}_TbaseST4tiny_AllSmmdH1_lr${lr}e${ep}_bs${batch_size}
DATA_DIR=${DATA_ROOT_DIR}
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}



mkdir -p $OUTPUT_DIR


python -u distill.py \
    --vocab_file $BERT_DIR/vocab.txt \
    --data_dir  $DATA_DIR \
    --bert_config_file_T $BERT_DIR/config.json \
    --bert_config_file_S $STUDENT_CONF_DIR/config.json \
    --tuned_checkpoint_T $trained_teacher_model \
    --init_checkpoint_S $student_init_model \
    --load_model_type bert \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length ${length} \
    --train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --ckpt_frequency 1 \
    --schedule slanted_triangular \
    --s_opt1 ${sopt1} \
    --output_dir $OUTPUT_DIR \
    --gradient_accumulation_steps ${accu} \
    --temperature ${temperature} \
    --output_att_sum false  \
    --output_encoded_layers true \
    --output_attention_layers false \
    --num_labels 2 \
    --matches L3_hidden_smmd \
              L3_hidden_mse \
    --predict_batch_size 128
