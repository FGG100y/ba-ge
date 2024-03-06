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

query = "你好! 请写出中国唐代诗人杜甫的关于茅草屋被秋风吹破的诗歌，并给出合适的英文翻译"
query = "你好! 请写出中国唐代诗人李白的《静夜思》全文，并给出合适的英文翻译"


#  async def producer(queue):
#      for _ in range(10):
#          sleep_time = random.randint(1, 2)
#          await queue.put(sleep_time)
#
#
#  async def consumer(queue):
#      while True:
#          sleep_time = await queue.get()
#          size = queue.qsize()
#          print(f'当前队列有：{size} 个元素')
#          url = f'http://httpbin.org/delay/{sleep_time}'
#          async with aiohttp.ClientSession() as client:
#              resp = await client.get(url)
#              print(await resp.json())
#
#  async def main():
#      queue = asyncio.Queue()
#      asyncio.create_task(producer(queue))
#      con = asyncio.create_task(consumer(queue))
#      await con
#
#
#  asyncio.run(main())


async def arun(client=client, query=query, verbose=False):
    print("Start querying LLM ...")

    queue = asyncio.Queue(maxsize=1)

    async def producer(q):
        response = ""
        for chunk in completion:
            text = chunk.choices[0].delta.content
            response += text if text else ""
            if verbose:
                print(text, end="", flush=True)
            if text[-1] in [".", "?", "。", "？"]:
                #  print(f"put into queue: {response}")
                await q.put(response)
                response = ""
                #  await q.put(text)

    async def consumer(q):
        while True:
            response = await q.get()
            tts_greeting(response, use_bark=False, xtts_sr=24000)


    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        #  model="mixtralMoE",  # also work like this.
        messages=[
            #  {"role": "system", "content": ""},  # mixtralMoE not support it
            {"role": "user", "content": query},
        ],
        stream=True,
    )

    asyncio.create_task(producer(queue))
    con = asyncio.create_task(consumer(queue))
    await con
    #  response = ""
    #  for chunk in completion:
    #      text = chunk.choices[0].delta.content
    #      response += text if text else ""
    #      print(text, end="", flush=True)


def run(client=client, query=query, verbose=False):
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

    streaming = True
    if streaming:
        asyncio.run(arun())
    else:
        txts = run(verbose=False)
