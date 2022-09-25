
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash run_infer_310.sh [MODEL_PATH] [TEST_PATH] [NEED_PREPROCESS]
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'."
  exit 1
fi

get_real_path() {
  if [ -z "$1" ]; then
    echo ""
  elif [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
model=$(get_real_path $1)
test_path=$(get_real_path $2)

if [ "$3" == "y" ] || [ "$3" == "n" ]; then
  need_preprocess=$3
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

echo "mindir name: "$model
echo "test_path: "$test_path
echo "need preprocess: "$need_preprocess

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest/
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

function preprocess_data() {
  if [ -d preprocess_310_Result ]; then
    rm -rf ./preprocess_310_Result
  fi
  mkdir preprocess_310_Result
  python ./preprocess_310.py --test_dir=$test_path --out_path=./preprocess_310_Result/
}

function compile_app() {
  cd ./ascend310_infer || exit
  bash build.sh &>build.log
}

function infer() {
  cp -r ./preprocess_310_Result/ ./ascend310_infer/
  cd - || exit
  if [ -d result_Files ]; then
    rm -rf ./result_Files
  fi
  mkdir result_Files

  ./ascend310_infer/build/DPTNet $model ./preprocess_310_Result/  &> infer.log

}

function cal_acc() {
  python ./postprocess.py --test_dir=$test_path --bin_path=./result_Files &> acc.log
}

if [ $need_preprocess == "y" ]; then
  preprocess_data
  if [ $? -ne 0 ]; then
    echo "preprocess dataset failed"
    exit 1
  fi
fi
compile_app
if [ $? -ne 0 ]; then
  echo "compile app code failed"
  exit 1
fi
infer
if [ $? -ne 0 ]; then
  echo " execute inference failed"
  exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
  echo "calculate accuracy failed"
  exit 1
fi
