原始项目：https://github.com/lipku/livetalking


本项目的优化点：
- 增加实时背景切换功能
- 对话界面优化
- 阿里云ASR支持

环境依赖见 requirements.freeze.txt


Real time interactive streaming digital human， realize audio video synchronous dialogue. It can basically achieve commercial effects.  
实时交互流式数字人，实现音视频同步对话。基本可以达到商用效果

## 为避免与3d数字人混淆，原项目metahuman-stream改名为livetalking，原有链接地址继续可用

## Features
1. 支持多种数字人模型: ernerf、musetalk、wav2lip
2. 支持声音克隆
3. 支持数字人说话被打断
4. 支持全身视频拼接
5. 支持rtmp和webrtc
6. 支持视频编排：不说话时播放自定义视频

## 1. Installation

Tested on Ubuntu 20.04, Python3.10, Pytorch 1.12 and CUDA 11.3

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda activate nerfstream
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
#如果不训练ernerf模型，不需要安装下面的库
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
pip install --upgrade "protobuf<=3.20.1"
```


## 2. Quick Start
默认采用ernerf模型，webrtc推流到srs  
### 2.1 运行srs
```
export CANDIDATE='<服务器外网ip>'
docker run --rm --env CANDIDATE=$CANDIDATE \
  -p 1935:1935 -p 8080:8080 -p 1985:1985 -p 8000:8000/udp \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 \
  objs/srs -c conf/rtc.conf
```

### 2.2 启动数字人：

```python
python app.py
```

如果访问不了huggingface，在运行前
```
export HF_ENDPOINT=https://hf-mirror.com
```

用浏览器打开http://serverip:8010/rtcpushapi.html, 在文本框输入任意文字，提交。数字人播报该段文字  
备注：服务端需要开放端口 tcp:8000,8010,1985; udp:8000

## 3. More Usage
使用说明: <https://livetalking-doc.readthedocs.io/>
  
## 4. Docker Run  
不需要前面的安装，直接运行。
```
docker run --gpus all -it --network=host --rm registry.cn-beijing.aliyuncs.com/codewithgpu2/lipku-metahuman-stream:vjo1Y6NJ3N
```
代码在/root/metahuman-stream，先git pull拉一下最新代码，然后执行命令同第2、3步 



## 5. 性能分析
1. 帧率  
在Tesla T4显卡上测试整体fps为18左右，如果去掉音视频编码推流，帧率在20左右。用4090显卡可以达到40多帧/秒。    
2. 延时  
整体延时3s左右  
（1）tts延时1.7s左右，目前用的edgetts，需要将每句话转完后一次性输入，可以优化tts改成流式输入  
（2）wav2vec延时0.4s，需要缓存18帧音频做计算 
（3）srs转发延时，设置srs服务器减少缓冲延时。具体配置可看 https://ossrs.net/lts/zh-cn/docs/v5/doc/low-latency
