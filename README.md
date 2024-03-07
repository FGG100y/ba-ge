# 名称

这是关于语音交互大语言模型的项目，目的就是构建一个有趣机器伙伴，我称之为“金坚”。

~~对了，就是那只最后投胎成了猫咪的、被小倩抛弃的、主人名叫宁采臣的小狗🐶 (gamgin) 。~~


# 项目初衷：

与其临渊羡鱼，不如退而结网。
原谅我这一生不羁放纵爱折腾。


注意1：
目前所有代码充其量就是一个“脚手架”（分支"scaffold"预计将会作为长期分支）

注意2：
一个参考：AMD7840 (32G RAM) & RXT4050 (6G) --> 迟钝的助手，而且相当容易就OOM😂。


# 项目组件：

![voice-assistant-arch](./images/voice-assistant-arch.png)

- 唤醒词：金坚 (为什么使用唤醒模式？唤醒词识别模型小，消耗不大）
- STT: ASR模型 -- whishper-large/faster-whisper
- LLM01: mixtral-7x8b-instruct/Yi-34b-chat.gguf -- llama.cpp server
- LLM02: Qwen-7b-chat/chatglm-6b -- 小一些的模型，可本地运行
- ~~TTS01: coqui-ai xtts -- limited 82 chars (400 tokens) to zh-cn (FIXME)~~
- TTS01: coqui-ai xtts -- 分而治之
- ~~TTS02: alternative: elevenlabs  -- 长度限制&收费~~
- TTS03: alternative: bark  -- 预训练模型；无长度限制 (考虑机器算力问题)
- TTS04: alternative: speechwhisper  -- (TODO) 预训练模型；无长度限制 (考虑机器算力问题)


# 怎么折腾

- 🙄 从语音输入到语音转文字到大模型生成再到文字转语音 （折腾不止）
    - [x] LLM->TTS: async/await 异步处理LLM生成和TTS文字转语音 （难言流畅（机器性能不足？但愿如此 😂 ））
    - [x] coquiai-xtts 开启deepspeed=True (最好有多点GPU内存，否则不能与faster-whisper共存)


# 做了些啥

- [x] 听话小管家🐶
    - [x] 唤醒词检测
    - [x] 唤醒词定制 (可去这里 porcupine 定制唤醒词)
    - [x] 唤醒流程挂后台

- [ ] 废话小管家🐶
    - [x] prototype (STT -> LLM -> TTS)
    - [ ] 语音转文字较快（延迟合适）
        - [ ] faster-whisper 据说内置了 VAD 算法，可折腾
    - [x] 解决中文语音合成长度限制问题 
        - 使用 elevenlabs 送来的是一只洋金坚🐶而且免费的不多😂
        - 使用 suna-bark 带来的也是一只洋金坚🐶而且相当耗费内存😂
        - 目前：coquiai-xtts分治法解决长度限制问题（你最好有一台好机器，或者好脾气）（极可折腾）

    - [ ] 音调、音色克隆 (洋金坚 -- 口音问题)
        - 解决思路：更好的中文演讲者音频文件

    - [ ] 大模型回复（延迟一般）
        - [x] 使用 stream=True 模式，缓解等待焦虑
        - [x] Mixtral-MoE/Yi-34b-chat.gguf 服务器最好有个24G显卡
        - [x] 本地部署更小参数量的模型 Qwen/chatglm)
        - [ ] tensorRT-llm triton server

    - [ ] 少点废话小管家🐶
        - [ ] LLM 提示工程
            - [ ] Naive RAG -- 讲个笑话乐呵乐呵啥的
                - 大模型拒绝讲成人笑话，作为一个工具它似乎很正经😂
                - 解决方案：uncensored-llm
            - [x] Naive RAG -- 朴素检索增强个人资料问答
                - [x] pdfs (unstructuredio 提供的工具需要更多算力 😂)
                - [x] txts (长文本断句需要改进)
            - [x] 使用内存向量 (numpy array, FAISS)
                - [x] FAISS
                - [ ] vectordb (pgvertor)
        - [ ] LLM 智能代理
            - [ ] functional-calling -- 使用api工具访问互联网
            - [ ] multi-step-search agent -- 综合搜索智能代理
            - [ ] multi-tasks agent: 播放本地音乐

- [ ] 干活小管家🐶
    - [ ] 语言交互设备控制
        - [ ] 家电开关控制
        - [ ] 家电调参控制
        - [ ] ...


# 其他事项

- Nvidia显卡环境变量 (可恶)：
```sh
# # nvcc --version 与 pytorch.__version__ 不匹配的问题：
export PATH="/usr/local/cuda-12.3/bin:$PATH"

# # libcudnn*.so.8 问题 (不用apt安装系统级依赖包，pytorch已打包这些依赖包)：
export LD_LIBRARY_PATH="/home/cll/.pyenv/versions/3.10.12/envs/vass-venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

