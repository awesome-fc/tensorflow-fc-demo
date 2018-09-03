# 函数计算部署机器学习遇到的问题和解法

随着 Serverless 的流行，将应用迁移到云上已经成了一种必然的趋势。
我们今天来看一下如何将机器学习应用迁移到[函数计算](fc.console.aliyun.com)上。

## 1. 本地开发

首先我们看一下本地开发机器学习应用的步骤。我们大概可以将本地开发概括为三个步骤，分别是代码编写，安装依赖，运行调试。我们分别来看一下。

### 1.1 代码编写

假定我们的项目结构为：

```
project root
├── index.py
├── model_data
│   ├── checkpoint
│   ├── model.data-00000-of-00001
│   ├── model.index
│   └── model.met
└── pic
    └── e2.jpg
```

其中 `index.py` 存放了机器学习相关的代码，`model_data` 存放了数据模型，pic 中存放了要进行测试的图片。


index.py 的内容为（代码参考了[这篇文章](https://www.jianshu.com/p/2498f1191ab4)）：

```python
# -*- coding:utf-8 -*-   
import os
import sys
import cv2  
import numpy as np
import tensorflow as tf  

saver = None

def reversePic(src):
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src[i, j] = 255 - src[i, j]
    return src 

def main():
    sess = tf.Session()  

    saver = tf.train.import_meta_graph('model_data/model.meta')
 
    saver.restore(sess, 'model_data/model')
    graph = tf.get_default_graph()
    
    input_x = sess.graph.get_tensor_by_name("Mul:0")
    y_conv2 = sess.graph.get_tensor_by_name("final_result:0")
    
    path="pic/e2.jpg"  
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    im = reversePic(im)

    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_CUBIC)  

    x_img = np.reshape(im , [-1 , 784])  
    output = sess.run(y_conv2 , feed_dict={input_x:x_img})  
    print 'the predict is %d' % (np.argmax(output)) 

    sess.close()

if __name__ == '__main__':  
    main()  
```

### 1.2 安装依赖
在运行应用前，需要先安装应用依赖的模块，这里主要依赖了 opencv 以及 tensorflow，安装方法很简单：

```bash
pip install opencv-python
pip install tensorflow
```

执行完这两条命令后，`opencv-python` 以及 `tensorflow` 就被安装到系统目录里。Linux 下默认为 /usr/local/lib/pythonX.Y/site-packages。

### 1.3 运行

运行时，python 会自动在配置的路径下查找相关模块并进行加载。

```bash
$ python index.py
the predict is 8
```

经过这三个步骤，我们就完成了本地机器学习应用的开发，我们接下来看下如何迁移应用到函数计算。

## 2. 迁移函数计算

### 2.1 本地开发与函数计算开发对比

首先，我们需要做一些准备工作。让我们来思考下函数计算应用开发方式与本地应用应用的开发方式有什么不同呢？

1. 代码入口。本地开发时，代码可以省略 main 函数，也可以提供 main 函数作为程序入口，但在函数计算中，函数入口是固定的，非 Http 触发器的函数入口必须是一个包含了两个参数的函数，比如：def handler(event, context)。

2. 模块依赖。本地开发时，项目依赖的模块通常会被安装到系统的某个目录。比如我们上面执行的 pip install tensorflow。而对于函数计算，由于为了能够最大限度的对应用进行优化，开放给用户的操作空间通常是比较小的。因此，对于函数计算，目前还无法做到安装项目依赖到运行环境。我们只能通过将自定义模块一同打包的方式。[参考](https://help.aliyun.com/document_detail/56316.html?spm=a2c4g.11186623.6.561.53781e91dGjcHE#attention)。

3. 运行。本地开发时，需要使用 python 命令或者 IDE 来运行代码。而在函数计算，我们需要首先部署应用到函数计算，再通过触发器或者控制台手动触发执行。

接下来我们针对这三点开发方式的不同对代码进行改造。

### 2.2 改造代码

#### 2.2.1 代码入口改造

这个比较简单，只需要将 

```python
def main():
```

修改为

```python
def handler(event, context):
```

并删除下面代码：

```python
if __name__ == '__main__':  
    main()  
```

#### 2.2.2. 模块依赖

这一块稍微复杂些。不同的语言因为模块加载机制的不同，这里的处理逻辑也会有差异。比如对于 java，无论是使用 maven，还是 gradle，都可以很容易的一键将代码以及依赖打包成 jar。但遗憾的是 python 目前没有这种机制。

我们先根据场景对 python 依赖的模块做个简单的分类。

> **应用依赖：** 对于本例中使用 pip 安装的模块，比如 `pip install tensorflow`，我们暂且将其称为应用依赖。
> **系统依赖：** 在某些场景下，python 安装的库仅仅是对底层 c、c++ 库调用的封装，例如使用 zbar 时，除了使用 pip install zbar，还要在系统中安装相应的库：apt-get install -y libzbar-dev。我们暂且把像 libzbar-dev 一样需要使用系统软件包管理器安装的库称为系统依赖。
> **资源依赖：** 对于一些应用，比如机器学习，启动后还需要加载数据模型，数据模型需要在程序启动时准备好，我们暂且将这种依赖称为资源依赖。资源依赖比较特殊，它是我们的应用逻辑所需要的，通常体积比较大。

对于应用依赖，我们可以通过 pip 的 `-t` 参数改变其安装位置，比如 `pip install -t $(pwd) tensorflow`。并且可以通过 `sys.path` 改变加载行为，使得可以从指定目录加载模块。

对于系统依赖，我们可以通过 `apt-get` 下载 deb 包，再利用 deb 包安装到指定目录。

```bash
apt-get install -y -d -o=dir::cache=$(pwd) libzbar-dev
for f in $(ls $(pwd)/archives/*deb); do dpkg -x $f $(pwd); done
rm -r archives
```

对于系统依赖包含的链接库，可以通过 `LD_LIBRARY_PATH` 变量改变其加载行为。

对于资源依赖，因为控制权在我们的代码里，因此只需要改变代码的处理逻辑就可以了。

根据上面的描述，我们可以整理成下面的表格：



| 类别 | 定义 | 安装方法举例 | 指定位置安装方法举例 | 影响加载的因素 |
| :--- | :--- | :--- | :--- | :--- |
| 应用依赖 | pip 安装的模块 | `pip install tensorflow` | `pip install -t $(pwd) tensorflow` | `sys.path` |
| 系统依赖 | 系统软件包管理器安装的依赖 | `apt-get install -y libzbar-dev` | `apt-get install -y -d -o=dir::cache=$(pwd) libzbar-dev`<br><br>`for f in $(ls $(pwd)/archives/*deb); do dpkg -x $f $(pwd); done`<br>`rm -r archives` | `LD_LIBRARY_PATH` |
| 资源依赖 | 代码依赖的资源，比如数据模型 | \ | \ | 由应用代码控制 |

#### 2.2.3 下载依赖的逻辑

对于我们的 demo 应用，存在两种依赖，一种是应用依赖，另一种是资源依赖。而需要我们特别处理的只有应用依赖。我们需要在项目目录下创建一个名为 applib 的目录，并下载应用依赖到该目录。这里需要注意的是如果引用的模块使用 C / C++ / go 编译出来的可执行文件或者库文件，那么推荐使用 fcli 的 [sbox](https://help.aliyun.com/document_detail/56316.html?spm=a2c4g.11186623.6.561.53781e91dGjcHE#adding-modules) 进行下载，使用方法为：

```bash
mkdir applib 
fcli shell
sbox -d applib -t python2.7
pip install -t $(pwd) tensorflow
pip install -t $(pwd) opencv-python
```

执行完毕后，就会发现 applib 中就包含了项目所需要的应用依赖。

#### 2.2.4 打包依赖上传到 OSS

机器学习的应用依赖、资源依赖通常比较大，会很容易超过函数计算对代码包的限制（50M）。为了避开这个问题，我们需要将这些依赖上传到 OSS：

```bash
cd applib && zip -r applib.zip * && mv applib.zip ../ ; cd ..
```

执行完毕后，项目会多出一个名为 applib.zip 的压缩包，上传到 oss 即可。

同样的，对资源依赖进行相同的操作：

```bash
cd model_data && zip -r model_data.zip * && mv model_data.zip ../ ; cd ..
```

#### 2.2.5 初始化依赖

这里我们提供一个模板代码，负责在函数第一次启动时，从 OSS 下载资源到本地、解压，并配置好相应的环境变量。我们可以在项目中创建一个名为 `loader.py` 文件，内容为：

```python
# -*- coding:utf-8 -*-   
import sys
import zipfile
import os
import oss2
import imp
import time

app_lib_object = os.environ['AppLibObject']
app_lib_dir = os.environ['AppLibDir']

model_object = os.environ['ModelObject']
model_dir = os.environ['ModelDir']

local = bool(os.getenv('local', ""))

print 'local running: ' + str(local)

inilized = False

def download_and_unzip_if_not_exist(objectKey, path, context):
    
    creds = context.credentials

    if (local):
        print 'thank you for running function in local!!!!!!'
        auth = oss2.Auth(creds.access_key_id,
                         creds.access_key_secret)
    else:
        auth = oss2.StsAuth(creds.access_key_id,
                            creds.access_key_secret,
                            creds.security_token)

    endpoint = os.environ['Endpoint']
    bucket = os.environ['Bucket']

    print 'objectKey: ' + objectKey
    print 'path: ' + path
    print 'endpoint: ' + endpoint
    print 'bucket: ' + bucket

    bucket = oss2.Bucket(auth, endpoint, bucket) 
    
    zipName = '/tmp/tmp.zip'

    print 'before downloading ' + objectKey + ' ...'
    start_download_time = time.time()
    bucket.get_object_to_file(objectKey, zipName)
    print 'after downloading, used %s seconds...' % (time.time() - start_download_time)

    if not os.path.exists(path):
        os.mkdir(path)

    print 'before unzipping ' + objectKey + ' ...'
    start_unzip_time = time.time()
    with zipfile.ZipFile(zipName, "r") as z:
        z.extractall(path)
    print 'unzipping done, used %s seconds...' % (time.time() - start_unzip_time)

def handler(event, context):
    global inilized
    if not inilized:
        if ( not local ):
            download_and_unzip_if_not_exist(app_lib_object, app_lib_dir, context) 
            download_and_unzip_if_not_exist(model_object, model_dir, context)
        sys.path.insert(1, app_lib_dir)
        print sys.path
        inilized = True

    file_handle, desc = None, None
    
    fn, modulePath, desc = imp.find_module('index')
    mod = imp.load_module('index', fn, modulePath, desc)

    request_handler = getattr(mod, 'handler')
    return request_handler(event, context)      
```

这段代码会首先读取 AppLibObject 环境变量，用于从 OSS 下载应用依赖，并解压到 AppLibDir 这个环境变量所代表的目录。

其次会读取 ModelObject 环境变量，用于从 OSS 下载资源依赖，并解压到 ModelDir 这个环境变量所代表的目录。

最后，当依赖准备妥当后，会调用 index.py 中的 handler 函数。

理论上，这个代码可以用于其它任何需要下载应用依赖、资源依赖的场景。而我们的 index.py 需要修改的，只有将原先的获取模型依赖的固定的路径，修改为利用 ModelDir 获取路径即可。

### 2.3 本地运行调试

代码编写完成后，我们的目录结构调整为：

```
project root
├── code
│   ├── index.py
│   ├── loader.py
│   └── pic
│       └── e2.jpg
├── applib
│.  └── *
└── model_data
    ├── checkpoint
    ├── model.data-00000-of-00001
    ├── model.index
    └── model.met
```

我们本地运行看下效果，这里我们借助于函数计算推出的 [fc-dcoker](https://github.com/aliyun/fc-docker) 工具。

为了避免本地每次调试做无谓的下载，我们取下巧，将应用依赖、资源依赖挂载到 `fc-docker` 中，并开启 `local` 的标识：

```bash
docker run --rm \
    -e local=true \
    -e AppLibObject=applib.zip \
    -e AppLibDir=/tmp/applib \
    -e ModelObject=model_data.zip \
    -e ModelDir=/tmp/model \
    -v $(pwd)/code:/code \
    -v $(pwd)/applib:/tmp/applib \
    -v $(pwd)/model_data:/tmp/model \
    aliyunfc/runtime-python2.7 \
   loader.handler
```

得到结果：

```bash
2018-08-28 17:44:16.043564: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supportsinstructions that this TensorFlow binary was not compiled to use: AVX2 FMA
FunctionCompute python runtime inited.
FC Invoke Start RequestId: f3ea930e-d7e2-4173-9726-453cdd89f18c
local running: True
['/code', '/tmp/applib', '/var/fc/runtime/python2.7/src', '/usr/local/lib/python27.zip', '/usr/local/lib/python2.7', '/usr/local/lib/python2.7/plat-linux2', '/usr/local/lib/python2.7/lib-tk', '/usr/local/lib/python2.7/lib-old', '/usr/local/lib/python2.7/lib-dynload', '/usr/local/lib/python2.7/site-packages']
2018-08-28T17:44:16.154Z f3ea930e-d7e2-4173-9726-453cdd89f18c [INFO] Restoring parameters from /tmp/model/model
the predict is 8

RequestId: f3ea930e-d7e2-4173-9726-453cdd89f18c          Billed Duration: 9954 ms        Memory Size: 1998 MB       Max Memory Used: 239 MB
```

### 2.4 部署

本地开发完成，接下来，我们就需要部署应用到线上了。这里我们借助函数计算推出的 [Fun](https://github.com/aliyun/fun) 工具。

Fun 工具使用步骤如下：

1. 去 [release](https://github.com/aliyun/fun/releases/tag/v2.2.1) 页面对应平台的 binary 版本，解压就可以使用。或者使用 `npm install @alicloud/fun -g` 也可以直接使用。
2. 使用 fun config 配置 ak、region 等信息。
3. 编写 template.yml
4. fun deploy 部署

是的，不需要登录控制台进行繁琐的配置，仅仅在项目下提供一个 template.yml 即可：

```yaml
ROSTemplateFormatVersion: '2015-09-01'
Transform: 'Aliyun::Serverless-2018-04-03'
Resources:
  tensorflow: # 服务名
    Type: 'Aliyun::Serverless::Service'
    Properties:
      Description: 'tensorflow demo'
      Policies: 
        - AliyunOSSReadOnlyAccess
    test: # 函数名
      Type: 'Aliyun::Serverless::Function'
      Properties:
        Handler: utils.handler
        CodeUri: ./code/
        Description: 'tensorflow application!'
        Runtime: python2.7
        MemorySize: 1024
        Timeout: 300
        EnvironmentVariables:
          Bucket: just-fc-test # 替换为自己的 oss bucket
          Endpoint: 'https://oss-cn-shanghai-internal.aliyuncs.com' # 替换掉 OSS Endpoint
          AppLibObject: applib.zip
          AppLibDir: /tmp/applib
          ModelObject: model_data.zip
          ModelDir: /tmp/model
```

至此，我们的项目中又多了一个 template.yml，结构为

```
project root
├── code
│   ├── index.py
│   ├── loader.py
│   └── pic
│       └── e2.jpg
├── applib
│   └── *
├── template.yml
└── model_data
    ├── checkpoint
    ├── model.data-00000-of-00001
    ├── model.index
    └── model.met
```

通过这一个 template.yml，执行 fun deploy 后即可创建好相应的服务、函数，并配置好函数的环境变量。

```bash
$ fun deploy
Waiting for service tensorflow to be deployed...
        Waiting for function test to be deployed...
                Waiting for packaging function test code...
                package function test code done
        function test deploy success
service tensorflow deploy success
```

即使修改了代码，只要重复执行 fun deploy 即可。

接下来，打开 `https://fc.console.aliyun.com/` 控制台，依次找到创建好的服务、函数，点击执行，即可得到与本地一致的输出：

![image](https://yqfile.alicdn.com/94290667130db3f5f13f1620796e5b68ce2c6522.png)


### 2.5 补充

在上面的例子中，我们只列出了应用依赖、资源依赖的情况。对于系统依赖的处理逻辑是比较简单的，比如我们拿 zbar 举例。除了需要在 applib 中通过 pip 安装 zbar，还要在 code 目录下新建一个 lib 目录，并通过 sandbox 在这个目录中执行：

```bash
apt-get install -y -d -o=dir::cache=$(pwd) libzbar-dev
for f in $(ls $(pwd)/archives/*deb); do dpkg -x $f $(pwd); done
rm -r archives
```

执行完成后，目录结构变化为：

```
project root
├── code
│   ├── lib
│   │   └── usr/lib
│   │       └── *

```

就像上面提到的，我们需要修改 LD_LIBRARY_PATH 来改变系统依赖的加载行为，因此我们需要在 template.yaml 中的 EnvironmentVariables 下添加一行：

```bash
LD_LIBRARY_PATH: /code/lib/usr/lib:/code:/code/lib:/usr/local/lib
```

至此，就可以直接在 index.py 等文件中直接使用 zbar 了。

## 3. 总结

结果一番努力，我们终于将机器学习应用上线到函数计算了。回顾上面的所有操作可以发现，其实大部分的改造工作都是可以通过工具解决的。无论是 template.yml，还是 loader.py 都是直接拿来就能用的。而真正需要开发者操作的也就只有下载依赖、修改对资源依赖的引用路径了。

如果您在迁移过程中遇到了困难，请加入我们的钉钉群，同时，我们还会为您提供针对平台优化的 tensorflow 给您：

![image](https://yqfile.alicdn.com/9bed3288ee8f16c5e0ef724d00fa1d7e81d2b2c8.png)


## 4. 参考

[函数计算使用自定义的模块](https://help.aliyun.com/document_detail/56316.html?spm=a2c4g.11186623.6.561.53781e91dGjcHE#adding-modules)
[函数计算安装依赖库方法小结](https://yq.aliyun.com/articles/602147)
[Tensorflow MINIST数据模型的训练，保存，恢复和手写字体识别](https://www.jianshu.com/p/2498f1191ab4)
[fc-docker](https://github.com/aliyun/fc-docker)
[Fun](https://github.com/aliyun/fun)


