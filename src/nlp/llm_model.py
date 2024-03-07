"""
NOTE 将 mixtral 换为 yi-34b-chat/qwen-chat 等中文模型，中文对话质量才有保证

# In Linux CLI (on a remote server):
```sh
# -ngl 0 means using CPU only for inference
alias SVRmixtral='./server -ngl 0 -c 8096 \
    -m ./models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
    --alias mixtralMoE'

cd /path/to/repo-of/llama.cpp/
# then run `SVRmixtral` which init mixtralMoE model using llama.cpp
```

# when the LLM is on the laptop (no remote server things):

- On laptop machine:
    just run: `python3 src/nlp/llm_model.py`

when the LLM is on a remote server:

- On laptop machine CLI:
    run1: `connect_llm_server PORT` (e.g., connect_llm_server 8080)
        where `connect_llm_server` is a bash function to forward localhost of
        remote server. So the script can make query to 'localhost:port'.

    run2: `python3 src/nlp/llm_model.py`

"""
import asyncio
import openai
from audio.utils.greeting import tts_greeting


client = openai.OpenAI(
    base_url="http://localhost:8080/v1",  # llama.cpp server
    api_key="sk-no-key-required"
)


async def arun(query, language, client=client, verbose=False):
    print("Start querying LLM ...")

    queue = asyncio.Queue(maxsize=2)

    # get completions from LLMs:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        #  model="mixtralMoE",  # also work like this.
        messages=[
            #  {"role": "system", "content": ""},  # mixtralMoE not support it
            {"role": "user", "content": query},
        ],
        stream=True,
    )

    asyncio.create_task(producer(queue, completion))
    consumer_task = asyncio.create_task(consumer(queue, language))
    await consumer_task


async def producer(q, completion, verbose=True):
    response = ""
    for chunk in completion:
        text = chunk.choices[0].delta.content
        if text is None:
            continue
        response += text if text else ""
        if verbose:
            print(text, end="", flush=True)

        # FIXME this is way too dummy conditions: (5/7 quatrains)
        if len(response) >= 12 and text[-1] in ["\n", ".", "。"]:  # , "?", "？"]:
            await q.put(response)
            response = ""  # reset the sentence. ortherwise it grow on and on

    # indicate the producer is done
    await q.put(None)


async def consumer(q, language):
    while True:
        response = await q.get()
        if response is None:
            break
        tts_greeting(response, use_bark=False, xtts_sr=24000, language=language)

        # notify the queue that the item has been processed
        q.task_done()


def run(query, client=client, verbose=False):
    print("Start querying LLM ...")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        #  model="mixtralMoE",  # also work like this.
        messages=[
            #  {"role": "system", "content": ""},  # mixtralMoE not support it
            {"role": "user", "content": query},
        ]
    )

    response = completion.choices[0].message.content
    if verbose:  # working as expected 2024-01-19
        print("user:", query)
        print("bot:", response)

    tts_greeting(response, use_bark=False, xtts_sr=24000)
    #  return response


if __name__ == "__main__":

    query = "你好! 请写出中国唐代诗人杜甫的关于茅草屋被秋风吹破的诗歌，并给出合适的英文翻译"
    query = "你好! 请写出中国唐代诗人李白的《静夜思》全文，并给出合适的英文翻译"
    LANGUAGE = "zh-cn"

    query = "who is the author of The Lord of the Ring?"
    LANGUAGE = "en"

    streaming = True
    if streaming:
        asyncio.run(arun(query))
    else:
        txts = run(query, verbose=True)

    breakpoint()
