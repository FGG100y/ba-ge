"""RAG using huggingface pipeline; Qwen-7b/Yi-6b for better Chinese understanding"""

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()
# Initialize global variables for use in functions
model_name_or_path = "models/hfLLMs/Mistral-7B-Instruct-v0.2"
#  OPENAI_API_KEY = "Empty"

persona = "东北著名狠人-范德彪"
llama_template = """
### [INST] Instruction: Answer the question based on your literature and art knowledge. Here is context to help, use it wisely:
{context}

Respond in the persona of %s

### QUESTION:
{question} [/INST]
"""
PROMPT = PromptTemplate(
    template=llama_template % (persona),
    input_variables=["context", "question"],
)



def run(query, vectordb, prompt=PROMPT):
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

    result_rag = llm_chain.invoke(query)
    print("\n>>", query)
    print(result_rag["result"], "\n")
    breakpoint()


def load_model_and_tokenizer(model_name_or_path):
    """
    Loads the model and tokenizer for text generation.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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


def load_and_process_document(file_path):
    """
    Loads and processes a document, returning a list of document chunks.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        # hyper-parameters
        chunk_size=500, chunk_overlap=50
    )
    docs = text_splitter.split_documents(pages)

    return docs


# FIXME 这个做法占用显存，应该先完成文档embedding，然后再需要的时候直接检索
def embed_documents(docs, language="en"):
    """
    Embeds document chunks using OpenAI's embeddings.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    try:
        modelPath = "models/hfLLMs/jina-embeddings-v2-base-zh"
        embeddings = HuggingFaceEmbeddings(  # langchain wrapper
            model_name=modelPath,
        )
        # storing embeddings in the FAISS
        vectordb = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        raise e

    return vectordb


def setup_retrieval_chain(text_generation_pipeline, prompt, vectordb):
    """
    Sets up the prompt template and LLM chain for text generation.
    """
    from langchain_community.llms.huggingface_pipeline import (
        HuggingFacePipeline,
    )

    llm = HuggingFacePipeline(
        pipeline=text_generation_pipeline,
        pipeline_kwargs={"do_sample": True, "temperature": 0.2},
    )
    llm_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return llm_chain


if __name__ == "__main__":
    #  docs = load_and_process_document("data/pdfs/en/llama2.pdf")
    docs = load_and_process_document("data/pdfs/zh/novel_最后一片藤叶.pdf")
    vectordb = embed_documents(docs, language="zh")
    question = "什么样的作品才能称为画家的杰作？"

    run(query=question, vectordb=vectordb)
