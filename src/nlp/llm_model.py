"""
In Linux CLI:

alias SVRmixtral='./server -ngl 0 -c 8096 \
    -m ./models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
    --alias mixtralMoE'

cd /path/to/repo-of/llama.cpp/;
run `SVRmixtral` which init mixtralMoE model using llama.cpp

01 when the LLM is on the laptop (no remote server things):

- On laptop machine CLI:
    just run: `python3 src/nlp/llm_model.py`

02 when the LLM is on a remote server:

- On laptop machine CLI:
    run1: `connect_llm_server PORT` (e.g., connect_llm_server 8080)
        where `connect_llm_server` is a bash function to forward localhost of
        remote server. So the script can make query to 'localhost:port'.

    run2: `python3 src/nlp/llm_model.py`

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
        #  model="mixtralMoE",  # also work like this.
        messages=[
            #  {"role": "system", "content": ""},  # mixtralMoE not support it
            {"role": "user", "content": query},
        ]
    )

    response = completion.choices[0].message.content
    if verbose:  # working as expected 2024-01-19
        print("usr:", query)
        print("bot:", response)

    return response


if __name__ == "__main__":
    query_llm(verbose=True)
