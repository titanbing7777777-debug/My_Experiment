python ../run_model.py -mode eval -model_checkpoint ../Models/asqp/allenai/tk-instruct-base-def-pos-asqp_check \
-experiment_name asqp_check -task asqp -output_path ../Output \
-inst_type 2 \
-id_tr_data_path ../Dataset/DiaASQ/train.csv \
-id_te_data_path ../Dataset/DiaASQ/test.csv \
-per_device_train_batch_size 16 -per_device_eval_batch_size 16 -num_train_epochs 4