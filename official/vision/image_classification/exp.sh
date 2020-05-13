# export TPU_NAME=lbt-tpu-europe
export DEVICE_COUNT=128
export TPU_NAME=v3-$DEVICE_COUNT

export STORAGE_BUCKET=gs://lbt-bucket-europe-west4
export MODEL_DIR=$STORAGE_BUCKET/large_imagenet
export DATA_DIR=$STORAGE_BUCKET/image_net

export SWITCH_FROM=4096
export SWITCH_TO=1024
export SWITCH_AT=50

per_replica_batch_size=`expr $SWITCH_TO / $DEVICE_COUNT`

if [[ $SWITCH_FROM != $SWITCH_TO ]]
then
  python3 classifier_trainer.py   \
  --mode=train_and_eval   \
  --model_type=resnet   \
  --dataset=imagenet   \
  --tpu=$TPU_NAME   \
  --model_dir=$MODEL_DIR/`expr $SWITCH_FROM`to`expr $SWITCH_TO`at`expr $SWITCH_AT`   \
  --data_dir=$DATA_DIR   \
  --config_file=configs/examples/resnet/imagenet/tpu.yaml \
  --params_override="train_dataset.builder=records,validation_dataset.builder=records,train_dataset.batch_size=$per_replica_batch_size"\
  --init_chkpt=$MODEL_DIR/$SWITCH_FROM/model.ckpt-00$SWITCH_AT \
  --SWITCH_FROM=$SWITCH_FROM \
  --SWITCH_TO=$SWITCH_TO
else
  python3 classifier_trainer.py   \
  --mode=train_and_eval   \
  --model_type=resnet   \
  --dataset=imagenet   \
  --tpu=$TPU_NAME   \
  --model_dir=$MODEL_DIR/$SWITCH_TO  \
  --data_dir=$DATA_DIR   \
  --config_file=configs/examples/resnet/imagenet/tpu.yaml \
  --params_override="train_dataset.builder=records,validation_dataset.builder=records,train_dataset.batch_size=$per_replica_batch_size"
fi

