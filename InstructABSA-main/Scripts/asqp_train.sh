python ../run_model.py -mode train -model_checkpoint ../tk-instruct-base-def-pos \
-experiment_name asqp_check -task asqp -output_dir ../Models \
-inst_type 1 \
-id_tr_data_path ../Dataset/DiaASQ/train.csv \
-id_te_data_path ../Dataset/DiaASQ/test.csv \
-id_val_data_path ../Dataset/DiaASQ/valid.csv \
-per_device_train_batch_size 16 -per_device_eval_batch_size 16 -num_train_epochs 4