"""与大语言模型进行语音聊天

voice chat with LLMs
(due to limited computation resources, say 32G RAM and RXT4050 6G)
NOTE faster-whisper-v3 + qwen-7b-chat-gptq-int4 --> GPU OOM error for a second round

"""
import asyncio
import openai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from audio import stt_model  # , tts_hfopt_bark, tts_elevenlabs, tts_model
from nlp import llm_model, rag_llm
from wake_word.wake_gamgin_stream import wake_gamgin
from utils import load_yaml_cfg as yamlcfg
from audio.utils.greeting import tts_greeting

# preset configurations:
preset_shorts = yamlcfg.load_config()
EXIT_WORDS_L = preset_shorts["tts_greetings"]["exit_words"]
HELLOS_D = preset_shorts["tts_greetings"]["hellos"]
GOODBYES_D = preset_shorts["tts_greetings"]["goodbyes"]
LLM_BACKUPS_D = preset_shorts["llm_backups"]

# TTS models:
# use coqui-ai xtts
#  XTTS_MODEL, CONFIG = tts_model.load_xtts_model()  # CPU is ok

# STT models:
faster_whisper = stt_model.load_faster_whisper()


def main(verbose=False):
    init = 1
    wake_again = False
    init_local_llm = 1

    while True:
        # PART01: wake word detection -------------------------------------------------

        if not init and wake_again:
            hello_again = HELLOS_D["hello_again"]
            tts_greeting(hello_again, use_bark=False, xtts_sr=24000)

        if init and wake_gamgin():
            # responding the calling:
            hello = HELLOS_D["say_hello"]
            tts_greeting(hello, use_bark=False, xtts_sr=24000)
            init = 0

        # PART02: speech2text (faster-whisper-large), input and transcribe ------------
        speech2text = stt_model.transcribe_fast(
            model=faster_whisper, duration=50, verbose=1
        )

        say_goodbye = [w for w in EXIT_WORDS_L if w in speech2text]
        if len(say_goodbye) > 0:
            intext = GOODBYES_D["say_goodbye"]
            tts_greeting(intext, use_bark=False, xtts_sr=24000)

            #  break  # 结束程序 (DEBUG only)

            # 如果用户说再见，则进入等待唤醒状态
            if wake_gamgin():
                wake_again = True
                continue

        # PART03: query the LLMs ------------------------------------------------------

        prompt = speech2text

        try:  # the BIGGER LLM on server first:
            asyncio.run(llm_model.arun(query=prompt, verbose=True))
            #  llm_model.run(query=prompt, verbose=True)
            init_local_llm = 0  # if server is fine, do not init local llm
        except (ConnectionRefusedError, openai.APIConnectionError):
            #  if init and init_local_llm:
            if init_local_llm:
                # LOCAL LLMs: embedding model and chat model
                embedding_model_name = "models/hfLLMs/m3e-large"
                embedding_model = HuggingFaceEmbeddings(
                    model_name=embedding_model_name
                )
                try:
                    vectordb = FAISS.load_local(
                        "./data/vectordb/faiss_index", embedding_model
                    )
                except Exception:
                    print("No local faiss index found. Create one now.")
                    docs = rag_llm.load_and_process_document(
                        "data/pdfs/zh/novel_最后一片藤叶.pdf"  # FIXME later
                    )
                    vectordb = rag_llm.embed_documents(docs, embedding_model)
                llm_chain = rag_llm.create_llm_chain(vectordb)
                init_local_llm = 0  # init locall llm once is enough

            try:  # the smaller LLM locally:
                llm_response = llm_chain.invoke(prompt)["result"]
                if verbose:  # working as expected 2024-01-19
                    print("usr:", prompt)
                    print("bot:", llm_response)
            except Exception as e:
                print(e)
                llm_response = LLM_BACKUPS_D[
                    "no_service"
                ]  # "无法连接到大模型服务，请稍后再试"

            # PART04: TTS of LLM result ---------------------------------------------------
            try:
                tts_greeting(llm_response, use_bark=False, xtts_sr=24000)
            except Exception as e:
                raise e


if __name__ == "__main__":
    main(verbose=True)
