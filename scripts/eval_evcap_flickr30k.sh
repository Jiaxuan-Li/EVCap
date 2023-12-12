SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
FLICKR_OUT_PATH=results/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL
mkdir -p $LOG_FOLDER

NOCAPS_LOG_FILE="$LOG_FOLDER/NOCAPS_${TIME_START}.log"

python -u eval_evcap.py \
--device cuda:$DEVICE \
--name_of_datasets flickr30k \
--path_of_val_datasets data/flickr30k/test_captions.json \
--image_folder data/flickr30k/flickr30k-images/flickr30k/image/ \
--out_path=$FLICKR_OUT_PATH \
|& tee -a  ${NOCAPS_LOG_FILE}


echo "==========================FLICKR30k EVAL================================"
python evaluation/cocoeval.py --result_file_path $FLICKR_OUT_PATH/flickr30k*.json |& tee -a  ${FLICKR_LOG_FILE}

