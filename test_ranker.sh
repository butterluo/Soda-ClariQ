CUDA_VISIBLE_DEVICES=0 python3 -m src.run_bert_ranker --do_test \
                                 --data_dir=./data --test_path=./data/test.tsv \
                                 --max_seq_len=60 \
                                 --log_dir=./log/log_ranker --output_dir=./output
