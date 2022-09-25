
if [ $# != 1 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "Usage: bash run_eval.sh [DEVICE_ID] [CKPT]  [VAL_PATH] [FILE_LIST] "
  echo "bash run_eval.sh 0 weights/gdprnn.ckpt  /dataset/"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export DEVICE_ID=$1
export RANK_SIZE=1

rm -rf ./eval
mkdir ./eval
cp -r ../*.py ./eval
cp -r ../src ./eval

env > env.log
python ./eval/evaluate.py  --device_id=$1 --model_path=$2 --data_dir=$3 > eval.log 2>&1 &