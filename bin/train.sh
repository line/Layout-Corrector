#!/usr/bin/env bash

DATASET=$1
EXPERIMENT=$2

if [ "${DATASET}" = "" ]; then
    echo "Please specify DATASET as the first args"
    exit;
fi

if [ "${EXPERIMENT}" = "" ]; then
    echo "Please specify EXPERIMENT as the second args"
    exit;
fi

ADDITIONAL_ARGS=""
if [ "${3}" != "" ]; then
    ADDITIONAL_ARGS="${ADDITIONAL_ARGS} ${@:3}"
fi
echo $ADDITIONAL_ARGS

NOW=$(date "+%Y%m%d%H%M%S")
VCPU=16

if [[ "$ADDITIONAL_ARGS" =~ "model.training_condition="([a-zA-Z0-9_]+) ]]; then
  CONDITION="${BASH_REMATCH[1]}"
else
  CONDITION="unconditional"
fi

DATA_DIR="./download/datasets"
JOB_DIR="tmp/jobs/${DATASET}/${EXPERIMENT}${OPTION}_${CONDITION}_${NOW}"
FID_WEIGHTS_DIR="./download/fid_weights/FIDNetV3"

echo "DATA_DIR=${DATA_DIR}"
echo "JOB_DIR=${JOB_DIR}"
echo "ADDITIONAL_ARGS=${ADDITIONAL_ARGS}"

SHARED_DEFAULT_ARGS="--multirun +experiment=${EXPERIMENT} fid_weight_dir=${FID_WEIGHTS_DIR} job_dir=${JOB_DIR} dataset=${DATASET} dataset.dir=${DATA_DIR} data.num_workers=${VCPU}"
if ! [[ "$ADDITIONAL_ARGS" =~ "seed=" ]]; then
  SEEDS=0
  SHARED_DEFAULT_ARGS="${SHARED_DEFAULT_ARGS} seed=${SEEDS}"
fi

python -m trainer.main ${SHARED_DEFAULT_ARGS} ${ADDITIONAL_ARGS}
