# 名称

这是关于语音交互大语言模型的项目，目的就是构建一个有趣机器伙伴，我称之为“金坚”。

对，就是那只最后投胎成了猫咪的、被小倩抛弃的、主人名叫宁采臣的小狗🐶 (gamgin) 。


# 项目主要组件：

![voice-assistant](./images/voice-assistant.png)

- 唤醒词：gamgin (为什么使用唤醒模式？唤醒词识别模型小，消耗不大）
- STT: ASR模型 -- whishper-large/faster-whisper
- LLM: mixtral-7x8b-instruct.gguf -- llama.cpp server
- TTS: coqui-ai xtts -- limited 82 chars (400 tokens) to zh-cn (FIXME)
- TTS alternative: elevenlabs  -- 长度限制宽松的很，但尚未有汉语纯正口音


# 主要功能：

支持非性能著称类机器本机离线运行。

- [x] 听话小管家🐶
    - [x] 唤醒词检测
    - [x] 唤醒词定制 (porcupine custom)
    - [x] 唤醒流程挂后台 (WakeBot(callback=callback))

- [ ] 废话小管家🐶
    - [x] pre-prototype (STT -> LLM -> TTS)
    - [x] 等待语音转文字，耐心等待大模型回答 （延迟严重）
    - [x] 语音转文字较快（延迟合适）
    - [x] 解决中文语音合成长度限制问题 (但😂 elevenlabs 送来的是一只洋金坚🐶)
    - [ ] 音调、音色克隆 (洋金坚 -- 口音问题)
    - [ ] 大模型快速回复（延迟合适）(需要更强计算能力支撑）

    - [ ] 少点废话小管家🐶
        - [ ] 播放本地音乐
        - [ ] RAG -- 个人图书问答
        - [ ] functional-calling -- 访问互联网

- [ ] 干活小管家🐶
    - [ ] 语言交互设备控制
        - [ ] 家电开关控制
        - [ ] 家电调参控制
        - [ ] ...


# 安装

```sh
python -m pip install gamgin
```


# 开发

```sh
# # nvcc --version 与 pytorch.__version__ 不匹配的问题：
# export PATH="/usr/local/cuda-12.3/bin:$PATH"

# # libcudnn*.so.8 问题 (不用apt安装系统级依赖包，pytorch已打包这些依赖包)：
# export LD_LIBRARY_PATH="/home/cll/.pyenv/versions/3.10.12/envs/vass-venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

