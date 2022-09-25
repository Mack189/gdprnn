
if [ $# != 1 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "bash run_standalone_train.sh [DEVICE_ID]"
  echo "bash run_standalone_train.sh 0"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export DEVICE_ID=$1
export RANK_ID=0
export RANK_SIZE=1
export SLOG_PRINT_TO_STDOUT=0


rm -rf ./train_gdprnn
mkdir ./train_gdprnn
cp -r ../*.py ./train_gdprnn
cp -r ../src/ ./train_gdprnn
cd ./train_gdprnn || exit
python train.py --device_id=$DEVICE_ID  > train.log 2>&1 &
