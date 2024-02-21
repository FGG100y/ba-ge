# 名称

这是关于语音交互大语言模型的项目，目的就是构建一个有趣机器伙伴，我称之为“金坚”。

对，就是那只最后投胎成了猫咪的、被小倩抛弃的、主人名叫宁采臣的小狗🐶 (gamgin) 。


# 项目主要组件：

![voice-assistant](./images/voice-assistant.png)

- 唤醒词：gamgin (为什么使用唤醒模式？唤醒词识别模型小，消耗不大）
- STT: ASR模型 -- whishper-large/faster-whisper
- LLM01: mixtral-7x8b-instruct/Yi-34b-chat.gguf -- llama.cpp server
- LLM02: Qwen-7b-chat/chatglm-6b -- 小一些的模型
- TTS01: coqui-ai xtts -- limited 82 chars (400 tokens) to zh-cn (FIXME)
- TTS02: alternative: elevenlabs  -- 长度限制&收费
- TTS03: alternative: bark  -- 预训练模型；无长度限制，本机器算力不足
- TTS04: alternative: speechwhisper  -- 预训练模型；无长度限制，本机器算力不足


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
    - [x] 大模型快速回复（延迟较高）(MoE/yi-34b 需要更强计算能力支撑)
    - [ ] 大模型快速回复（延迟合适）(本地部署更小参数量的模型 Qwen/chatglm)

    - [ ] 少点废话小管家🐶
        - [ ] LLM 提示工程
            - [ ] Naive RAG -- 讲个笑话乐呵乐呵啥的
            - [ ] Naive RAG -- 朴素检索增强个人资料问答
            - [ ] 使用内存向量 (numpy array, FAISS); 再考虑 vectordb (pgvertor)
        - [ ] LLM 智能代理
            - [ ] functional-calling -- 使用api工具访问互联网
            - [ ] multi-step-search agent -- 综合搜索智能代理
            - [ ] multi-tasks agent: 播放本地音乐

- [ ] 干活小管家🐶
    - [ ] 语言交互设备控制
        - [ ] 家电开关控制
        - [ ] 家电调参控制
        - [ ] ...


# 安装 & 开发

- PyPI 安装：
```sh
python -m pip install gamgin
```

- N卡环境变量：
```sh
# # nvcc --version 与 pytorch.__version__ 不匹配的问题：
# export PATH="/usr/local/cuda-12.3/bin:$PATH"

# # libcudnn*.so.8 问题 (不用apt安装系统级依赖包，pytorch已打包这些依赖包)：
# export LD_LIBRARY_PATH="/home/cll/.pyenv/versions/3.10.12/envs/vass-venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

