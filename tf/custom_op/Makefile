TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CCFLAGS = -D_GLIBCXX_USE_CXX11_ABI=0

custom_op.so: custom_op.cc
	echo $(TF_INC)
	g++ -std=c++11 -shared custom_op.cc -o custom_op.so -fPIC -I $(TF_INC) $(CCFLAGS) -O2

all:
	custom_op.so
