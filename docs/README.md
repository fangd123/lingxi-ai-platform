# 海马轻帆深度学习平台

## 环境安装

首次使用时配置

### 前提条件

## miniconda3安装

基于阿里云的miniconda3镜像

Anaconda安装文件路径：https://mirrors.aliyun.com/anaconda/miniconda/

请从中选择最新的合适当前系统的文件版本进行安装。

### miniconda3配置

#### 初始化环境

1. windows 系统

进入“开始”菜单->"miniconda3"->"Anaconda Prompt (miniconda3)"，输入：

```shell
conda init powershell
```
即可在今后的`powershell`终端中，直接使用conda了


2. Linux 系统

进入`conda`默认安装目录后，执行下列语句

```bash
cd bin
conda init
```

如果使用的是非Bash的shell，则需要手动进行以下操作：

进入`$HOME/.bashrc`中，将

```bash
# >>> conda initialize >>>
# <<< conda initialize <<<
```
之间的内容拷贝下来，粘贴至目前使用的Shell的配置文件的结尾，执行`source`语句激活配置。

完成上述操作后，在终端的最左侧将会有一个`(base)`出现，若没有，则说明配置为生效，请咨询同事。


#### 镜像加速

由于网络原因，默认的安装仓库连接速度缓慢，因此需要使用国内镜像仓库

Linux用户可以通过修改用户目录下的`.condarc`文件，默认位置为：`$HOME/.condarc`

Windows 用户无法直接创建名为`.condarc`的文件，可执行下列命令生成对应文件,默认位置为：`$USER/.condarc`

```bash
conda config --set show_channel_urls yes
```

清除`.condarc`文件中所有的内容，复制如下内容：

```yaml
channels:
  - defaults
show_channel_urls: true
default_channels:
  - http://mirrors.aliyun.com/anaconda/pkgs/main
  - http://mirrors.aliyun.com/anaconda/pkgs/r
  - http://mirrors.aliyun.com/anaconda/pkgs/msys2
custom_channels:
  conda-forge: http://mirrors.aliyun.com/anaconda/cloud
  msys2: http://mirrors.aliyun.com/anaconda/cloud
  bioconda: http://mirrors.aliyun.com/anaconda/cloud
  menpo: http://mirrors.aliyun.com/anaconda/cloud
  pytorch: http://mirrors.aliyun.com/anaconda/cloud
  simpleitk: http://mirrors.aliyun.com/anaconda/cloud
```

即可添加 Anaconda Python 免费仓库。

配置完成可运行 `conda clean -i` 清除索引缓存。

### Pypi镜像配置

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

### GPU 环境创建（基于Anaconda）

环境要求：
脚本会创建一个名为`lingxi_dl`的conda环境

```bash
conda deactivate
conda env create -f create_env/env-GPU.yml
conda activate lingxi_dl
pip install -r create_env/requirements-GPU.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### CPU环境(模型开发和代码调试)

~~**MacOS**先执行`brew install libomp`~~

~~**Linux**先执行`sudo apt-get install libgomp1`~~

```bash
conda deactivate
conda env create -f create_env/env-CPU.yml
conda activate lingxi_dl
pip install -r create_env/requirements-CPU.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 数据和模型管理

该项目默认使用DVC模块，用于对模型和数据进行管理，可参考[DVC指南](https://dvc.org/doc/start/data-versioning)

执行脚本`bash init_dvc.sh`初始化DVC工具（若使用当前模板，则可跳过该命令）

将下面两行命令添加至环境变量中
```.env
# 最好将下面两个变量设置添加至环境变量中
export AWS_SECRET_ACCESS_KEY="IO1UOMtGn4eN5rzGNzOGKPOU6n8metxq"
export AWS_ACCESS_KEY_ID="minio"
```

1. 数据和模型上传

不论是添加和修改文件都需要执行下面的命令

```bash
dvc add 模型或者数据文件
git add 对应生成的*.dvc文件（具体可参见上条命令的末尾说明）
dvc commit
git commit -am "提交说明"
dvc push
git push
```

2. 数据和模型拉取
```bash
git pull
dvc pull
```

3. 数据和模型的删除
```bash
dvc remove 需要删除的对应的*.dvc文件
dvc commit
git commit -am "提交说明"
dvc push
git push
```
4. 数据和模型的版本切换
```bash
git checkout <...>
dvc checkout
```

### 训练数据格式

注：数据中确保不存在空数据的问题，以及出现`nan`这样的字段，否则的话，分词的时候会处理错误
#### 分类问题

数据表头为

* `index	sentence	label`

数据和表头中间用`\t`分隔

*train.tsv*
```text
index	sentence	label
0	石豹子提着两个酒坛大笑着回来。	0
1	方汀汀被董冬冬抱住整个人都软了,她死死扣住他的脖子,凑上去闻了闻。	0
2	周春红没找到擦手布,举着湿漉漉的双手,扭身面朝朱朝阳,指指口袋。	0
3	秦川拽起胡经,一把将他推在最前面。	0
```

*dev.tsv*
```text
index	sentence	label
0	石豹子提着两个酒坛大笑着回来。	0
1	方汀汀被董冬冬抱住整个人都软了,她死死扣住他的脖子,凑上去闻了闻。	0
2	周春红没找到擦手布,举着湿漉漉的双手,扭身面朝朱朝阳,指指口袋。	0
3	秦川拽起胡经,一把将他推在最前面。	0
```

*test.tsv*
```text
index	sentence	label
0	石豹子提着两个酒坛大笑着回来。	0
1	方汀汀被董冬冬抱住整个人都软了,她死死扣住他的脖子,凑上去闻了闻。	0
2	周春红没找到擦手布,举着湿漉漉的双手,扭身面朝朱朝阳,指指口袋。	0
3	秦川拽起胡经,一把将他推在最前面。	0
```

#### NER问题

请将将数据文件存储至`data/ner`目录中

数据分为两列，第一列为单字，第二列为标签，中间用` `（空格）隔开，不同条数据之间用**空行**隔开

*train.txt*

```text
东 B-说话人
郭 I-说话人
宇 I-说话人
露 O
出 O
无 O
奈 O
的 O
苦 O
笑 O
： O
" O
如 O
果 O
我 O
说 O
我 O
是 O
担 O
心 O
那 O
只 O
半 O
妖 O

在 O
队 O
员 O
们 O
下 O
车 O
了 O
以 O
后 O
， O
何 B-说话人
云 I-说话人
教 I-说话人
练 I-说话人
对 O
大 O
家 O
说 O
道 O
： O
" O
这 O
之 O
后 O
我 O
就 O
没 O
法 O
陪 O
着 O
你 O
们 O
了 O
， O
诸 O
位 O
多 O
努 O
力 O
吧 O
. O
. O
. O
. O
. O
. O
噗 O
嗤 O
。 O
" O
```

在`lingxi_ner.py`文件中，修改下列注释位置
```python
def _info(self):
    return datasets.DatasetInfo(
        description=_DESCRIPTION,
        features=datasets.Features(
            {
                "id": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "ner_tags": datasets.Sequence(
                    datasets.features.ClassLabel(
                        # 此处修改NER标签
                        names=[
                            "O",
                            "B-说话人",
                            "I-说话人",
                        ]
                    )
                ),
            }
        ),
        supervised_keys=None,
        homepage="https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily",
    )
```

**数据集记得及时使用DVC命令进行版本控制！！！**

## 训练

基于huggingface的transformers的example修改

### 训练参数介绍

见`模型参数.md`文件说明

注：训练参数中

* `--do_train` 是进行训练，在`output_dir`中会生成`all_results.json`文件记录训练结果
* `--do_eval` 是进行测试，在`output_dir`中会生成`all_results.json`文件记录训练结果
* `--do_predict` 是进行预测，会在`output_dir`中生成一份`predict_results`开头的预测文件，其中包含预测结果和对应的概率值

这三个参数可以单独使用，也可以共同使用。


### 模型运行结果可视化

除了命令行会输出一份模型的结果之外

在网站：[http://wandb.hmqf.lo](http://wandb.hmqf.lo)也会保留一份项目结果，便于分享

账号：hmqf@hmqf.com
密码：hmqf1234

### 分类

参考链接：[text-classification](https://github.com/huggingface/transformers/tree/master/examples/text-classification)

训练脚本`run.sh`

* 模型默认使用存储在NAS上剧本小说预训练BERT模型作为初始化模型
* 训练结果默认存储在`result`文件夹中

### NER

训练脚本`run_ner.sh`

若不使用NAS上的模型，则需要自行下载模型，并将模型路径替换`run.sh`脚本中的`hfl/chinese-bert-wwm-ext`

模型下载方法：

从[chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main)
下载模型文件，所有文件都要下载，然后将下载后的路径替换上面的路径

## 结果分析

NER模型结果分析

## 模型优化

**注：以下过程必须进行评价指标的比较！！！**

### ONNX导出和优化

根据当前任务类型，选择`optimize/export_onnx_cls.py`或者`optimize/export_onnx_ner.py`脚本进行ONNX文件导出，

具体命令如下:
```bash
python optimize/export_onnx_cls.py [模型文件所在的文件夹路径] [输出的ONNX文件路径]
```

再编辑`optimize/optimize.sh`中的`raw.onnx`名字替换为导出的ONNX文件名，执行`bash optimize/optimize.sh`命令，对模型进行图融合和FP16导出

### 小模型训练

若当前任务比较简单，可以尝试使用小模型直接进行训练和优化，小模型的预训练模型路径为`/nfs/protech/模型库/预训练模型/script_novel_pretrain_bert_rbt3`,
该预训练模型已经使用小说/剧本文本数据进行二次预训练，优先使用该模型。

其他的小模型路径为：

* `/nfs/protech/模型库/预训练模型/rbt3_script_output`
* `/nfs/protech/模型库/预训练模型/rbtl3_script_output`
* `/nfs/protech/模型库/预训练模型/rbt4_script_output`
* `/nfs/protech/模型库/预训练模型/rbt6_script_output`

可以自行尝试不同模型，选择效果最好的

小模型的对应参数修改

* num_train_epochs 5 // 酌情增加 
* per_device_train_batch_size 64 // 酌情增加


~~### 模型蒸馏~~

~~使用哈工大讯飞联合实验室出品的`TextBrewer`作为模型蒸馏工具~~

~~官方示例：[notebook_examples](https://github.com/airaria/TextBrewer/tree/master/examples/notebook_examples)~~

~~参考链接：[TextBrewer](http://textbrewer.hfl-rc.com/)~~


### 模型量化

参考：https://onnxruntime.ai/docs/performance/quantization.html

模型量化过程主要将权重转换为INT8，在最终指标损失0.1~0.3%的基础上，提升模型推理速度，目前只能用于CPU服务器，GPU服务部署无法使用

使用方法：

执行`optimize/quantize.py`命令，例如
```bash
python optimize/quantize.py [需要量化的模型路径] [量化后的ONNX文件路径]
```

### ONNX 模型评测

对于优化后模型，需要进行评测，得出每次优化后的评价指标的变化，用于最终的模型决策

使用方法：

执行`optimize/evaluate_onnx_cls.py`（该脚本用于分类模型的指标评测），例如

```bash
python optimize/evaluate_onnx_cls.py [测试数据集文件路径] [模型生成预测结果文件路径] [ONNX文件路径]
```


## 部署

以下操作均在`deploy`文件夹中完成

**注意** 

为了兼顾模型性能和最终处理的准确性，设计了两种文本预处理模式

* 截断：直接将文本超出最大长度126字的内容进行截断，不处理超出部分的内容
* 超长文本：待处理的单行文本字数超过设定的最大长度126字，但是又希望能够处理超出部分的长度的内容，从中提取相关实体

分类任务默认使用**截断**模式，NER任务默认使用**超长文本**模式

若需要对此进行修改，需要在数据进入预处理步骤前，控制文本的长度

### 本地部署

分类任务：

1. 修改`deploy/api/config.py`文件，根据注释填写对应变量
2. 进入`deploy/api/routers/classify.py`文件
3. 修改`@router.post('/', response_model=ClassifyResult)`的`/`处内容，作为接口路径

NER任务：

1. 修改`deploy/api/config.py`文件，根据注释填写对应变量
2. 进入`deploy/api/routers/ner.py`文件
3. 修改如下内容：

```python
## 修改为此次使用的BIO实体
label_list = ['O', 'B-说话人', 'I-说话人  ']
# 根据接口文档，修改接口路径
@router.post('/novel/shr', response_model=NerResult)
```



### Docker 部署

以后再写

## API接口

### CLS 分类接口

#### Request

`type`和`debug`参数一般省略不写，如果写了可能会有问题，具体用法参考代码

```python
import requests
import json

url = "0.0.0.0:8000/oneline"

payload = json.dumps({
  "type":"chi",
  "debug":False,
  "texts": [
    "今天我上街"
  ]
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
```

## NER 实体识别接口

#### Request

同CLS接口

#### Response
- Body
```json
{
    "speakers": [
        []
    ],
    "roles": [
        [
            {
                "start": 0,
                "end": 3,
                "text": "方文达"
            }
        ]
    ]
}
```

## 日志和监控

日志配置文件为`deploy/logging_config.json`，默认的日志文件名为`access.log`，可以根据需要对日志配置进行修改

监控接口采用`Prometheus`完成，配置步骤如下：

1. 在对应的`prometheus.yml`的`job_name: 'dl_mode'`中，添加当前服务的IP和端口地址
2. 在`grafana`界面中，添加当前服务对应的监控界面

