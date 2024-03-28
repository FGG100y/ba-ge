"""RAG using huggingface pipeline; Qwen-7b/Yi-6b for better Chinese understanding

gpustat (peak usage):
[0] NVIDIA GeForce RTX 4050 Laptop GPU | 54 ℃,  99 % |  5890 /  6141 MB | python3/148030(?M)

"""

import torch
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

#  from auto_gptq import exllama_set_max_input_length

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize global variables for use in functions
model_name_or_path = "models/hfLLMs/Qwen1.5-7B-Chat-GPTQ-Int4"
#  model_name_or_path = "models/hfLLMs/Mistral-7B-Instruct-v0.2"

apply_persona = False
#  apply_persona = True
if apply_persona:
    persona = "东北著名狠人-范德彪"
    llama_template = """
    ### [INST] Instruction: Answer the question based on your literature and art knowledge. Here is context to help, use it wisely:
    {context}

    Respond in the persona of %s

    ### QUESTION:
    {question} [/INST]
    """
else:
    llama_template = """
    ### [INST] Instruction: Answer the question based on your literature and art knowledge. Here is context to help, use it wisely:
    {context}

    ### QUESTION:
    {question} [/INST]
    """

# 2024-03-01 01:04 Fri
# NOTE NOTE NOTE 使用如下的中文系统提示 (<|im_start|>system) 则模型无法进行角色扮演!
#  你是人工智能助手，根据你对文学和艺术的知识进行问题回答或解释，不知道的就回答不知道，不要捏造假信息来回答。请合理利用以下的背景知识:
#  尝试扮演 %s 这个人物性格来回答
if apply_persona:
    chatml_template = """
    <|im_start|>system
    Answer the question based on your literature and art knowledge. Here is context to help, use it wisely:
    {context}

    Respond in the persona of %s

    <|im_end|>
    <|im_start|>user
    {question}
    <|im_end|>
    <|im_start|>assistant
    """
else:
    chatml_template = """
    <|im_start|>system
    Answer the question based on your literature and art knowledge. Here is context to help, use it wisely:
    {context}
    <|im_end|>
    <|im_start|>user
    {question}
    <|im_end|>
    <|im_start|>assistant
    """

using_chatml = True
if using_chatml:
    if apply_persona:
        PROMPT = PromptTemplate(
            template=chatml_template % (persona),
            input_variables=["context", "question"],
        )
    else:
        PROMPT = PromptTemplate(
            template=chatml_template,
            input_variables=["context", "question"],
        )
else:
    if apply_persona:
        PROMPT = PromptTemplate(
            template=llama_template % (persona),
            input_variables=["context", "question"],
        )
    else:
        PROMPT = PromptTemplate(
            template=llama_template,
            input_variables=["context", "question"],
        )


# TODO add streaming mode (https://github.com/langchain-ai/langchain/issues/2918)
def create_llm_chain(vectordb, prompt=PROMPT):
    """
    Main function to run the modularized code.
    """
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    text_generation_pipeline = create_text_generation_pipeline(
        model, tokenizer
    )

    llm_chain = setup_retrieval_chain(
        text_generation_pipeline, prompt, vectordb
    )

    return llm_chain


def load_model_and_tokenizer(model_name_or_path):
    """
    Loads the model and tokenizer for text generation.
    """
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    if "gptq" in model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device,
            #  attn_implementation="flash_attention_2",
        )
        #  model = exllama_set_max_input_length(model, max_input_length=4096)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            #  attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

    return model, tokenizer


def create_text_generation_pipeline(model, tokenizer):
    """
    Creates a text generation pipeline using the specified model and tokenizer.
    """
    from transformers import pipeline

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        repetition_penalty=1.1,
        max_new_tokens=1000,
    )

    return text_generation_pipeline


def load_and_process_document(
    file_path, directory_path=None, use_unstructure=True, from_scratch=False
):
    """
    Loads and processes a document, returning a list of document chunks.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import (PyPDFLoader,
                                                      UnstructuredFileLoader)

    if use_unstructure:
        if from_scratch:
            # TODO load zh pdfs using nlp_utils .unstructrue and .zh_sentence_split
            from nlp.nlp_utils import unstructed_pdfs as utlis_unstructed
            from nlp.nlp_utils import zh_sentence_split as utlis_split

            contents, tables = utlis_unstructed.load_and_split(directory_path)
            texts_l = [docs[0][i].text for i in range(len(contents))]
            # FIXME not finish implementing
            breakpoint()
        else:
            #  loader = UnstructuredFileLoader(file_path, mode="elements")
            loader = UnstructuredFileLoader(file_path)
            pages = loader.load()
            for page in pages:
                page.page_content = page.page_content.replace("\n", "")
            #  breakpoint()
    else:
        # FIXME no good for chinese docs
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        # hyper-parameters
        chunk_size=100,
        chunk_overlap=20,
    )
    docs = text_splitter.split_documents(pages)

    return docs


def embed_documents(docs, embedding_model):
    """
    Embeds document chunks using OpenAI's embeddings."""
    from langchain_community.vectorstores import FAISS

    try:
        # storing embeddings in the FAISS
        vectordb = FAISS.from_documents(docs, embedding_model)
    except Exception as e:
        raise e
    else:
        vectordb.save_local("./data/vectordb/faiss_index")

    return vectordb


def setup_retrieval_chain(text_generation_pipeline, prompt, vectordb):
    """
    Sets up the prompt template and LLM chain for text generation.
    """
    from langchain_community.llms.huggingface_pipeline import \
        HuggingFacePipeline

    llm = HuggingFacePipeline(
        pipeline=text_generation_pipeline,
        pipeline_kwargs={"do_sample": True, "temperature": 0.1},
    )
    llm_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return llm_chain


if __name__ == "__main__":
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    docs = load_and_process_document("data/pdfs/zh/novel_最后一片藤叶.pdf")
    question = "为什么最后一片常春藤叶不会掉落？"
    question = "贝尔曼的杰作是什么？"

    #  docs = load_and_process_document("data/pdfs/en/llama2.pdf")
    #  breakpoint()
    #  question = "how to train llama2 effectively?"

    rebuild_vdb = True
    embedding_model_name = "models/hfLLMs/m3e-large"
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    if not rebuild_vdb:
        print("Load local faiss index.")
        vectordb = FAISS.load_local(
            "./data/vectordb/faiss_index", embedding_model
        )
    else:
        print("No local faiss index. Create one now.")
        docs = load_and_process_document(
            "data/pdfs/zh/novel_最后一片藤叶.pdf"  # FIXME later
        )
        vectordb = embed_documents(docs, embedding_model)

    llm_chain = create_llm_chain(vectordb=vectordb)
    result_rag = llm_chain.invoke(question)  # TODO 怎么知道它用了什么prompt？
    print("\n>>", question)
    print(result_rag["result"], "\n")
    breakpoint()
