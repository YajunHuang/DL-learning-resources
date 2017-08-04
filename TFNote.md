# My Tensorflow Tutorial

## Overview
TensorFlow relies on a highly efficient C++ backend to do its computation. The connection to this backend is called a session. The common usage for TensorFlow programs is to first create a graph and then launch it in a session.


```
import numpy as np
import tensorflow as tf

# A custom model 
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss= loss,
      train_op=train)
 
estimator = tf.contrib.learn.Estimator(model_fn=model)

# define our input data set
x=np.array([1., 2., 3., 4.])
y=np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# input_fn can be replaced by a custom input_dn
def my_input_fn():
  # Preprocess your data here...
  # ...then return 1) a mapping of feature columns to Tensors with
  # the corresponding feature data, and 2) a Tensor containing labels
  return feature_cols, labels
	
# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))
	
```

[Convert data to TensorFlow format example.](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py)

MetaGraph contains the information required to continue training, perform evaluation, or run inference on a previously trained graph.

## Source code learning
### Suggestion 1
源码我推荐几个python目录下非常值得看的基础类定义：[Ref](https://www.zhihu.com/question/41667903) [DjangoPeng](https://github.com/DjangoPeng)

- framework/Ops.py：定义了Tensor、Graph、Opreator类等
- Ops/Variables.py：定义了Variable类


『深度长文』Tensorflow代码解析:

- [『深度长文』Tensorflow代码解析（一）](http://chuansong.me/n/1589265951023)
- [『深度长文』Tensorflow代码解析（二）](http://chuansong.me/n/1613722651323)


TF系统开发使用了bazel工具实现工程代码自动化管理，使用了protobuf实现了跨设备数据传输，使用了swig库实现python接口封装。

Tensorflow核心框架使用C++编写，API接口文件定义在tensorflow/core/public目录下，主要文件是tensor_c_api.h文件，C++语言直接调用这些头文件即可。


## Reading data
There are three main methods of getting data into a TensorFlow program:

- Feeding: Python code provides the data when running each step.
- Reading from files: an input pipeline reads the data from files at the beginning of a TensorFlow graph.
- Preloaded data: a constant or variable in the TensorFlow graph holds all the data (for small data sets).

### Input and Readers
TensorFlow provides a set of Reader classes for reading data formats. There are defined in file:

*/python/ops/io_ops.py*

## SaveModel


### The Tensorflow data store ecosystem [Blog](https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125)

Graph metadata: .meta file. The meta chkp files hold the compressed Protobufs graph of your model and all the metadata associated (collections, learning rate, operations, etc.)

Variables data: .data file. The chkp files holds the data (weights) itself (this one is usually quite big in size).

The checkpoint file is just a bookkeeping file that you can use in combination of high-level helper for loading different time saved chkp files.

### Model Export

Tensorflow 训练的结果可以使用某种格式导出，如 SessionBundle、SavedModel、FreezedGraph 等等。目前Tensorflow on Pai 平台支持将使用 GenericSignature 方式导出的 SessionBundle 发布到 PAI 在线预测服务。

[__SessionBundle__](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/session_bundle/README.md): tensorflow.contrib.session_bundle.exporter. 

[__SavedModel__](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model): tensorflow.python.saved_model. SavedModel provides a language-neutral format to save machine-learned models that is recoverable and hermetic. It enables higher-level systems and tools to produce, consume and transform TensorFlow models.

SavedModel manages and builds upon existing TensorFlow primitives such as [TensorFlow Saver](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/training/saver.py) and MetaGraphDef. Specifically, SavedModel wraps a TensorFlow Saver. The Saver is primarily used to generate the variable checkpoints. SavedModel will replace the existing [TensorFlow Inference Model Format(Session Bundle)](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/session_bundle/README.md) as the canonical way to export TensorFlow graphs for serving.

[__Freezing Graph__](https://www.tensorflow.org/extend/tool_developers/): One confusing part about this is that the weights usually aren't stored inside the file format during training. Instead, they're held in separate checkpoint files, and there are Variable ops in the graph that load the latest values when they're initialized. It's often not very convenient to have separate files when you're deploying to production, so there's the [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) script that takes a graph definition and a set of checkpoints and freezes them together into a single file.

What this does is load the GraphDef, pull in the values for all the variables from the latest checkpoint file, and then replace each Variable op with a Const that has the numerical data for the weights stored in its attributes It then strips away all the extraneous nodes that aren't used for forward inference, and saves out the resulting GraphDef into an output file. [TensorFlow: How to freeze a model and serve it with a python API](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc)



## Serving


## tf.contrib.learn
TensorFlow’s high-level machine learning API (tf.contrib.learn) makes it easy to configure, train, and evaluate a variety of machine learning models. It is written by python to process data, build model, etc.

### Input function
Question: How to cope with data unbalance problem in _input\_fn()_ ?

### Estimator

Question: how to build customized estimator ?

## Python Wrapper
TensorFlow uses **SWIG** to wrap the C++ backend code. For example in the I/O module */python/lib/io/file_io.py*:

```
File IO methods that wrap the C++ FileSystem API.
The C++ FileSystem API is SWIG wrapped in file_io.i. These functions call those
to accomplish basic File IO operations.
```

## Keras
Being able to go from idea to result with the least possible delay is key to doing good research.

## Problems


## Tensorflow On PAI
[docs](http://gitlab.alibaba-inc.com/algo/pai-tensorflow-doc/tree/master)

https://www.atatech.org/activity/105
https://www.atatech.org/articles/78501/?flag_data_from=mail_daily_recommend&uid=144789

https://www.atatech.org/articles/78435/?flag_data_from=mail_daily_recommend&uid=144789
