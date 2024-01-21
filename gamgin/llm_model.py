"""
On server side:

alias SVRmixtral='./server -ngl 0 -c 8096 \
    -m ./models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
    --alias mixtralMoE'

cd /path/to/repo-of-llama.cpp/;
run `SVRmixtral` which init mixtralMoE model using llama.cpp

On laptop machine:
run `connect_llm_server PORT` (e.g., connect_llm_server 8080)
where `connect_llm_server` is a bash function to connect to remote server.

"""
import openai


client = openai.OpenAI(
    base_url="http://localhost:8080/v1",  # llama.cpp server; mixtralMoE
    api_key="sk-no-key-required"
)

query = "Hello! 请写出中国唐代诗人李白的《静夜思》全文，并给出合适的英文翻译"


def query_llm(client=client, query=query, verbose=False):
    print("Start querying LLM ...")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        #  model="mixtralMoE",  # not work like this.
        messages=[
            #  {"role": "system", "content": ""},  # mixtralMoE not support it
            {"role": "user", "content": query},
        ]
    )

    response = completion.choices[0].message.content
    if verbose:  # working as expected 2024-01-19
        print("user:", query)
        print("bot:", response)

    return response


if __name__ == "__main__":
    query_llm(verbose=True)
