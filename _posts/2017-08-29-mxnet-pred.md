---
layout: post
title: MXNet Prediction C++ 接口使用
date: 2017-08-28 22:10:00.000000000 +00:00
categories: MXNet C++
---

## 需求
最近在用C++实现一个CV算法框架。实现的时候对不同的特征提取方式做了抽象：基本的图像特征抽取类、
光流特征抽取类、或者CNN深度特征抽取类。其中，CNN的参数是通过MXNet的Python接口训练的，希望实现
一个C++版本的CNN特征抽取，于是就需要用到MXNet的C++接口。具体来说，就是如何用C++代码实现用OpenCV
读取图片，把图片feed进网络然后取出网络的inference结果的过程。全程参考自官方的[example](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/predict-cpp)。

## 调用C++接口
#### BufferFile类
用于读取训练完的参数`*.params`和符号`*.json`
{% highlight c++ %}
class BufferFile {
 public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            length_ = 0;
            buffer_ = NULL;
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... "<< length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        if (buffer_) {
          delete[] buffer_;
          buffer_ = NULL;
        }
    }
};
{% endhighlight %}
这个类很简单，作用就是无视文件格式，把文件读入内置的`buffer_`成员变量中，通过`GetLength()`和
`GetBuffer()`方法可以获取读入的数据的长度和内容。

### Pred Handle:
这个是与MXNet交互的界面，通过这个handle我们可以设置要输入到网络的数据，可以获取相应的输出。

####  Pred Handle的初始化，最核心的就是下面这个API：
{% highlight c++ %}
MXPredCreate(
     static_cast<const char*>json_data.GetBuffer(),
     static_cast<const char*>param_data.GetBuffer(),
     static_cast<size_t>(param_data.GetLength()),
     DEV,
     DEV_ID,
     num_input_node,
     input_keys,
     input_shape_indptr,
     input_shape_data,
     &pred_hnd);
{% endhighlight %}

对这里的变量做一下说明：

* `json_data`就是用模型的json文件初始化的`BufferFile`对象：
    {% highlight c++ %}json_data = BufferFile("/path/to/net-symbol.json");{% endhighlight %}
* `param_data`：同上，只不过是用模型参数文件初始化的：
    {% highlight c++ %}param_data = BufferFile("/path/to/net-0000.param");{% endhighlight %}
* `DEV`与`DEV_ID`：DEV等于2表示GPU，等于1表示CPU；DEV_ID表示用哪一个DEV（0~4）
* `num_input_node`：表示网络的输入个数，对于一个简单的分类网络，就是1，表示只有一个图像图像输入，
    如果是像R-CNN这样的网络，还会有第二个输入是ROIPooling层所需的ROI，这样就需要把node数量设置成2。
    总之，`num_input_node`和具体的网络有关
* `input_keys`：一个指向字符串指针数组的指针，数组的每个元素都是一个字符串，对应网络的输入节点的名字，比如一个网络有
    两个输入节点：`input1, input2 = mx.sym.Variable('data1'), mx.sym.Variable('data2')`，相应的`input_keys`如下：
{% highlight c++ %}
const char* input_key[2] = {"data1", "data2"};
const char** input_keys = input_key;
{% endhighlight %}
* `input_shape_data`：是表示输入数据大小的元素为`mx_uint`类型的数组，举个例子，上面的`data1`和`data2`都是10x3x224x224的图像，
    即一个batch有10张3通道的224*224大小的图像，那么`input_shape_data`应该如下：
{% highlight c++ %}
    const mx_uint input_shape_data[8] = {
        10, 3, 224, 224,    // data1的大小
        10, 3, 224, 224     // data2的大小
    };
{% endhighlight %}
* `input_shape_indptr`：表示`input_shape_data`各个维度和输入的对应关系，它也是一个`mx_uint`类型的数组，
    数组的大小是网络输入节点个数加一。`input_shape_indptr[i+1] - input_shape_indptr[i]`表示第i个输入
    在`input_shape_data`里面对应的尺寸数据的长度，`input_shape_indptr[i]`是第i个输入的尺寸数据在
    `input_shape_data`中开始的位置。比如：
{% highlight c++ %}
    const mx_uint input_shape_indptr[3] = {
        0,  // data1的尺寸信息从input_shape_data的第0个位置开始, 到4结束: 10, 3, 224, 224
        4,  // data2的尺寸信息从inptu_shape_data的第4个位置开始, 到8结束: 10, 3, 224, 224
        8};
{% endhighlight %}
* `pred_hnd`：MXNet预测handle，类型为`PredictorHandle`

#### 把图像放进网络
假设我们的模型接收1 x 3 x 224 x 224的图像作为输入，假设现有一个待输入的`Mat`类型图像，如何把
它放进网络里呢？MXNet接收的输入实际上一块连续的内存，大小由`input_shape_data`指定，所以我们就需要
把`Mat`转换成内存中的一块区域，实际的存储顺序如下：
```
|--------B--------|--------G--------|--------R--------|
|<---- W x H ---->|<---- W x H ---->|<---- W x H ---->|
```
即按通道顺序存储，每个通道内按照列优先的顺序存，于是可以实现一个如下所示的转换函数：
{% highlight c++ %}
// 假设内存空间已经预先分配到pData所指向的内存
void cvtMat2MXData(const Mat& img, float* pData, float mean_r = 0.0f, float mean_g = 0.0f, float mean_b = 0.0f){
    int w = img.cols, h = img.rows;
    // 检查是否符合网络的输入要求
    CHECK_EQ(w, INPUT_IMG_WIDTH);
    CHECK_EQ(h, INPUT_IMG_HEIGHT);
    int size = w * h;
    float* ptr_im_b = pData, *ptr_im_g = pData + size;, *ptr_im_r = pData + size + size;
    for(int i = 0; i < h; ++i){
        auto ptr = img.ptr<uchar>(i);
        for(int j = 0; j < w; ++j){
            // 顺便进行减均值的操作
            *ptr_im_b++ = static_cast<float>(*ptr++) - mean_b;
            *ptr_im_g++ = static_cast<float>(*ptr++) - mean_g;
            *ptr_im_r++ = static_cast<float>(*ptr++) - mean_r;
        }
    }
}
{% endhighlight %}
经过上面的变换，通过下面这个API就可以把图像送进网络：
{% highlight c++ %}
MXPredSetInput(pred_hnd, "data", pData, 3 * INPUT_IMG_WIDTH * INPUT_IMG_HEIGHT);
{% endhighlight %}
实际上，我们只需要把数据按照模型的输入的尺寸按顺序放入内存，再调用这个接口，就可以把各种数据放进网络。
比如，R-CNN的roi输入是2维的：N x 5，N是ROI个数，5是指`(image_index, x1, y1, x2, y2)`，于是，
对应的内存区域应该如下：
```
|-------BOX-------|-------BOX-------|-------BOX-------| ...
|<------ 5 ------>|<------ 5 ------>|<------ 5 ------>| ...
```

#### 获取输出
设置好输入后，需要调用`MXPredForward(pred_hnd)`来进行前向传播。通过`MXPredGetOutput`接口可以获取
网络的输出：
{% highlight c++ %}
MXPredGetOutput(pred_hnd, out_ind, pOutput_data, OUTPUT_SIZE);
{% endhighlight %}
其中，`out_ind`是整数，即获取第几个输出，如果网络只有一个输出，它就是0；pOutput_data是一个指向
`mx_float`类型数组的指针，大小由`OUTPUT_SIZE`确定。`pOutput_data`的内存排列方式和输入是
一样的，即按照最右侧维度优先的方式排列。

#### 编译链接
借助CMake可以非常方便地完成编译连接。直接在CMakeLists.txt中加入MXNet的头文件路径和链接库名称和位置即可：

```
link_directories(/path/to/mxnet/lib) # path where libmxnet.so can be found
add_executable( my_program
    main.cpp
    ...
)
target_link_libraries( my_program
    ${OPENCV_LIBS}
    ...
    mxnet                   # just 'mxnet', since the shared lib is named 'libmxnet.so'
)
target_include_directories( my_program
    ./include
    ...                     # other includes
    /path/to/mxnet/include  # mxnet include
)
```
写完CMakeLists.txt后运行`cmake /path/to/CMakeLists.txt; make`即可。
