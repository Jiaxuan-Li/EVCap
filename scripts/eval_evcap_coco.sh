SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
NOCAPS_OUT_PATH=results/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL
mkdir -p $LOG_FOLDER

NOCAPS_LOG_FILE="$LOG_FOLDER/NOCAPS_${TIME_START}.log"

python -u eval_evcap.py \
--device cuda:$DEVICE \
--name_of_datasets coco \
--path_of_val_datasets data/coco/test_captions.json \
--image_folder data/coco/coco2014/val2014/ \
--out_path=$NOCAPS_OUT_PATH \
|& tee -a  ${NOCAPS_LOG_FILE}

echo "==========================COCO EVAL================================"
python evaluation/cocoeval.py --result_file_path $NOCAPS_OUT_PATH/coco*.json |& tee -a  ${COCO_LOG_FILE}

