#! /bin/bash



if [[ -z $MNIST_ML_ROOT ]]; then 
    echo "MNIST_ML_ROOT is not set. Please set it to the root directory of the MNIST_ML project."
    exit 1
fi


dir=$( echo "$1" | tr "[:lower:]" "[:upper:]" )
echo dir = $dir
model_name_lower=$(echo "$dir" | tr "A-Z" "a-z" )
echo low = $model_name_lower
mkdir -p $MNIST_ML_ROOT/lib/$dir/inc $MNIST_ML_ROOT/lib/$dir/src

touch $MNIST_ML_ROOT/$dir/CMakeLists.txt
touch $MNIST_ML_ROOT/lib/$dir/inc/"$model_name_lower.hh"
touch $MNIST_ML_ROOT/lib/$dir/src/"$model_name_lower.cc"

echo "add_subdirectory(\${LIB_DIR}/$dir)" >> $MNIST_ML_ROOT/cmake/targets.txt
 