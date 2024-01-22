# 名称

这是关于语音交互大语言模型的项目，目的就是构建一个有趣机器伙伴，我称之为“金坚”。

对，就是那只最后投胎成了猫咪的、被小倩抛弃的、主人名叫宁采臣的小狗🐶 (gamgin) 。


# 安装

```sh
python -m pip install gamgin
```


# 项目主要模块：

![voice-assistant](./images/voice-assistant.png)

- 唤醒词：gamgin (为什么使用唤醒模式？唤醒词识别模型小，消耗不大）
- STT: ASR模型 -- whishper-large
- LLM: mixtral-7x8b-instruct.gguf -- llama.cpp server
- TTS: pyttxs3/espeak?/coqui-ai xtts


# 主要功能：

- [ ] 听话小管家🐶 -- User: 喂，gamgin.  Bot: 咩啊啊啊啊？
    - [x] 唤醒词检测
    - [ ] 唤醒词定制
    - [ ] 唤醒流程挂后台

- [ ] 废话小管家🐶
    - [x] pre-prototype (STT -> LLM -> TTS)
    - [ ] 低延迟问答

- [ ] 少点废话小管家🐶
    - [ ] 语言交互设备控制
        - [ ] 家电开关控制
        - [ ] 家电调参控制
        - [ ] ...


