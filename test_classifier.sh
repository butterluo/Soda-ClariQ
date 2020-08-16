CUDA_VISIBLE_DEVICES=0 python3 -m src.run_bert_classifier --do_test \
                                 --data_dir=./data --test_path=./data/test.tsv \
                                 --bert_model_dir=./pretrain/bert/base-uncased \
                                 --log_dir=./log/log_classifier --output_dir=./output
