#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=4

RESUME=$1
ARCH=${2-mobilenetv3_large}

DATASET=${3-pitts}
SCALE=${4-30k}

if [ $# -lt 1 ]
  then
    echo "Arguments error: <MODEL PATH>"
    echo "Optional arguments: <ARCH (default:vgg16)> <DATASET (default:pitts)> <SCALE (default:250k)>"
    exit 1
fi

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
test.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
    --test-batch-size 4 -j 2 \
    --width 640 --height 480 --features 4096 \
    --vlad --reduction \
    --resume ${RESUME}
    # --sync-gather
