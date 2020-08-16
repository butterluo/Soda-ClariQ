CUDA_VISIBLE_DEVICES=0 python3 -m src.run_bert_classifier --do_train \
                                 --data_dir=./data --bert_model_dir=./pretrain/bert/base-uncased --max_seq_len=50 \
                                 --log_dir=./log/log_classifier --batch_size=12 --num_epochs=10
