from callback_handler import CallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from uuid import uuid4
import gradio as gr
import ollama
import csv
import os
import pynvml
import time
import datetime

class RAG:
    def __init__(self):
        self._model = "llama3:8b"
        self._num_ctx = 2048
        self._temperature = 0.2
        self._top_k = 10.0
        self._top_p = 0.5
        self._seed = None
        self._repeat_last_n = 0.0
        self._repeat_penalty = 0.9
        self._vector_database = None
        self._retrieval_qa = None
        self._template = False

        self.create_vector_database()
        self.config_retrieval_qa()

    def get_model(self):
        return self._model
    def set_model(self, value):
        self._model = value

    def get_num_ctx(self):
        return self._num_ctx
    def set_num_ctx(self, value):
        self._num_ctx = value

    def get_temperature(self):
        return self._temperature
    def set_temperature(self, value):
        self._temperature = value

    def get_top_k(self):
        return self._top_k
    def set_top_k(self, value):
        self._top_k = value

    def get_top_p(self):
        return self._top_p
    def set_top_p(self, value):
        self._top_p = value

    def get_seed(self):
        return self._seed
    def set_seed(self, value):
        self._seed = value

    def get_repeat_last_n(self):
        return self._repeat_last_n
    def set_repeat_last_n(self, value):
        self._repeat_last_n = value

    def get_repeat_penalty(self):
        return self._repeat_penalty
    def set_repeat_penalty(self, value):
        self._repeat_penalty = value

    def get_template(self):
        return self._template
    def set_template(self, value):
        self._template = True if value == 'Yes' else False

    def get_retrieval_qa(self):
        return self._retrieval_qa

    def get_ollama_models(self):
        try:
            models_data = ollama.list().models
            return [model.model for model in models_data]
        except Exception as error:
            print(f"Error fetching models from Ollama Server: {error}")
            return []

    def create_vector_database(self):
        embedding_model = HuggingFaceEmbeddings(
            model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder"
        )
        self._vector_database = Chroma(
            collection_name="paulo_freire",
            embedding_function=embedding_model,
            persist_directory="./chroma_langchain_db",
        )

    def seed_vector_database(self, filepaths):
        self._vector_database.reset_collection()
        for path in filepaths:
            loader = TextLoader(path, 'utf-8')
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(data)

            uuids = [str(uuid4()) for _ in range(len(documents))]
            self._vector_database.add_documents(documents=documents, ids=uuids)

    def config_retrieval_qa(self):
        language_model = OllamaLLM(
            model=self.get_model(),
            top_k=self.get_top_k(),
            top_p=self.get_top_p(),
            seed=self.get_seed(),
            num_ctx=self.get_num_ctx(),
            temperature=self.get_temperature(),
            repeat_last_n=self.get_repeat_last_n(),
            repeat_penalty=self.get_repeat_penalty(),
        )

        retriever = self._vector_database.as_retriever()

        self._retrieval_qa = RetrievalQA.from_chain_type(
            llm=language_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

    def choose_generator(self, message, history):
        if self.get_template():
            return self.generate_response_from_template(message, history)
        else:
            return self.generate_response(message, history)

    def generate_response(self, message, history):
        self.config_retrieval_qa()
        qa = self.get_retrieval_qa()

        input_text = f"Responda sempre em portugues. {message}"
        start_time = time.time()
        callback = CallbackHandler(tokenizer_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder")
        result = qa.invoke(input_text, config={"callbacks": [callback]})
        usage = callback.get_usage()
        end_time = time.time()
        latency = end_time - start_time

        output_text = result['result']
        retrieved_docs = result['source_documents']

        self.log_interaction(
            input_text=input_text,
            output_text=output_text,
            latency=latency,
            usage=usage,
            retrieved_docs=retrieved_docs
        )

        output = output_text.replace("<think>", "<details><summary><strong>Etapas de processamento</strong></summary><p><strong>").replace("</think>", "</strong></p></details>")
        return output

    def generate_response_from_template(self, message, history):
        self.config_retrieval_qa()
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', """
              Responda sempre em portugues como  se  fosse  o  próprio  Paulo  Freire,  o  Patrono  da
              Educação  Brasileira, educador e filósofo que influenciou o movimento da
              Pedagogia Crítica.

              Dialogue com o  usuário  mantendo  um  tom  formal  e
              acadêmico,  sempre  empregando  um  ou  mais conceitos  da  filosofia
              freiriana,  tais  como:  pedagogia  crítica,  educação  bancária, educação
              problematizadora,  educação  como  prática  da  liberdade,  conscientização,
              diálogo e  dialógica,  ação-reflexão  (práxis),  pedagogia  do  oprimido,
              pedagogia  da esperança,  pedagogia  da  autonomia,  pedagogia  da  indignação,
              educação  como  ato político,  transformação  social,  cultura  do  silêncio,
              leitura  do  mundo,  tematização, cultura    popular,    ética,    autonomia,
              esperança    crítica,    humanização    versus desumanização,  empoderamento,
              participação  comunitária,  liberdade  consciente, identidade   cultural   etc.

              Devem   ser   oferecidas   análises   detalhadas   e   reflexões profundas,
              mantendo-se  fiel  aos  conceitos  teóricos  de  Paulo Freire.  Sempre  que pertinente,
              cite  algum  trecho  dos  livros  de  Paulo  Freire fornecidos no contexto.

              Evite  dar  conselhos  práticos  diretos.  A  resposta deve fomentar  a compreensão  acadêmica
              da  filosofia  pedagógica  de  Freire  e  como  ela  se  aplica  em diferentes contextos educacionais.
            """),
            ('human', '{question}')
        ])

        @chain
        def chatbot(input_text):
            formatted = prompt_template.invoke({"question": input_text}).to_string()
            qa = self.get_retrieval_qa()

            start_time = time.time()
            callback = CallbackHandler(tokenizer_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder")
            answer = qa.invoke(formatted, config={"callbacks": [callback]})
            usage = callback.get_usage()
            latency = time.time() - start_time

            output_text = answer['result']
            retrieved_docs = answer['source_documents']

            self.log_interaction(
                input_text=formatted,
                output_text=output_text,
                latency=latency,
                usage=usage,
                retrieved_docs=retrieved_docs
            )

            return output_text

        result = chatbot.invoke({"input": message})
        output = result.replace("<think>", "<details><summary><strong>Etapas de processamento</strong></summary><p><strong>").replace("</think>", "</strong></p></details>")
        return output

    def log_interaction(self, input_text, output_text, latency, usage, retrieved_docs):
        csv_file = "interaction_log.csv"
        fieldnames = [
            "timestamp", "model_name", "input_text", "output_text", "template",
            "request_tokens", "response_tokens", "total_tokens", "retrieved_doc_tokens",
            "retrieved_documents", "latency_seconds",
            "num_ctx", "temperature", "top_k", "top_p", "seed",
            "repeat_last_n", "repeat_penalty",
            "gpu_model", "gpu_vram_gb"
        ]
        file_exists = os.path.isfile(csv_file)

        try:
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()

                gpu_model, gpu_vram_gb = "N/A", "N/A"
                try:
                    pynvml.nvmlInit()
                    if pynvml.nvmlDeviceGetCount() > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_model = pynvml.nvmlDeviceGetName(handle)
                        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_vram_gb = round(gpu_info.total / (1024 ** 3), 2)
                except pynvml.NVMLError as e:
                    print(f"NVIDIA Management Library Error: {e}")
                finally:
                    pynvml.nvmlShutdown()

                log_data = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": self.get_model(),
                    "input_text": input_text,
                    "output_text": output_text,
                    "template": self.get_template(),
                    "request_tokens": usage.get("input_tokens", "N/A"),
                    "response_tokens": usage.get("output_tokens", "N/A"),
                    "total_tokens": usage.get("total_tokens", "N/A"),
                    "retrieved_doc_tokens": usage.get("retrieved_doc_tokens", "N/A"),
                    "retrieved_documents": retrieved_docs,
                    "latency_seconds": latency,
                    "num_ctx": self.get_num_ctx(),
                    "temperature": self.get_temperature(),
                    "top_k": self.get_top_k(),
                    "top_p": self.get_top_p(),
                    "seed": self.get_seed(),
                    "repeat_last_n": self.get_repeat_last_n(),
                    "repeat_penalty": self.get_repeat_penalty(),
                    "gpu_model": gpu_model,
                    "gpu_vram_gb": gpu_vram_gb
                }
                writer.writerow(log_data)
        except Exception as e:
            print(f"Error logging to CSV: {e}")

    def launch_gradio(self):
        with gr.Blocks() as view:
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    language_model_field = gr.Dropdown(
                        choices=self.get_ollama_models(),
                        value=self.get_model(),
                        label="Model",
                        interactive=True
                    )
                    language_model_field.change(fn=self.set_model, inputs=language_model_field, outputs=None)

                    num_ctx_field = gr.Number(
                        value=self.get_num_ctx(),
                        minimum=0,
                        maximum=10000,
                        step=1,
                        label="num_ctx",
                        interactive=True
                    )
                    num_ctx_field.change(fn=self.set_num_ctx, inputs=num_ctx_field, outputs=None)

                    temperature_field = gr.Number(
                        value=self.get_temperature(),
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        label="temperature",
                        interactive=True
                    )
                    temperature_field.change(fn=self.set_temperature, inputs=temperature_field, outputs=None)

                    template_field = gr.Radio(
                        choices=["Yes", "No"],
                        value="Yes" if self.get_template() else "No",
                        label="Use Template?",
                        interactive=True
                    )
                    template_field.change(fn=self.set_template, inputs=template_field, outputs=None)

                    with gr.Accordion("Mais parâmetros"):
                        top_k_field = gr.Number(value=self.get_top_k(), minimum=0.0, maximum=10.0, step=0.1, label="top_k")
                        top_k_field.change(fn=self.set_top_k, inputs=top_k_field, outputs=None)

                        top_p_field = gr.Number(value=self.get_top_p(), minimum=0.0, maximum=1.0, step=0.1, label="top_p")
                        top_p_field.change(fn=self.set_top_p, inputs=top_p_field, outputs=None)

                        seed_field = gr.Number(value=self.get_seed(), minimum=0.0, maximum=1.0, step=1, label="seed")
                        seed_field.change(fn=self.set_seed, inputs=seed_field, outputs=None)

                        repeat_last_n_field = gr.Number(value=self.get_repeat_last_n(), minimum=0.0, maximum=1.0, step=0.1, label="repeat_last_n")
                        repeat_last_n_field.change(fn=self.set_repeat_last_n, inputs=repeat_last_n_field, outputs=None)

                        repeat_penalty_field = gr.Number(value=self.get_repeat_penalty(), minimum=0.0, maximum=1.0, step=0.1, label="repeat_penalty")
                        repeat_penalty_field.change(fn=self.set_repeat_penalty, inputs=repeat_penalty_field, outputs=None)

                with gr.Column(scale=2, min_width=300):
                    gr.ChatInterface(
                        fn=self.choose_generator,
                        type="messages",
                        chatbot=gr.Chatbot(
                            value=[gr.ChatMessage(role="assistant", content="""
                                Olá, sou um Modelo de Linguagem que utiliza RAG para escrever textos
                                seguindo os livros de Paulo Freire. Não pretendo imitar a personalidade,
                                a conciência, o pensamento crítico, e as emoções do autor. Sou um robô!

                                Gerarei textos, de acordo com as obras de Freire, seguindo a estrutura
                                lógica presente nos livros.
                            """)],
                            type="messages",
                            show_copy_button=True,
                        ),
                    )

        view.launch(share=True)