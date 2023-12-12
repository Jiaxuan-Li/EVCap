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
--name_of_datasets nocaps \
--path_of_val_datasets data/nocaps/nocaps_corpus.json \
--image_folder data/nocaps \
--out_path=$NOCAPS_OUT_PATH \
|& tee -a  ${NOCAPS_LOG_FILE} 


echo "==========================NOCAPS IN-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/indomain*.json |& tee -a  ${NOCAPS_LOG_FILE}
echo "==========================NOCAPS NEAR-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/neardomain*.json |& tee -a  ${NOCAPS_LOG_FILE}
echo "==========================NOCAPS OUT-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/outdomain*.json |& tee -a  ${NOCAPS_LOG_FILE}
echo "==========================NOCAPS ALL-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/overall*.json |& tee -a  ${NOCAPS_LOG_FILE}


