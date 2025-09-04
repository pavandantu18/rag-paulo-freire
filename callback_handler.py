from langchain_core.callbacks import BaseCallbackHandler
from transformers import AutoTokenizer

class CallbackHandler(BaseCallbackHandler):
    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.input_tokens = 0
        self.output_tokens = 0
        self.retrieved_doc_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        for prompt in prompts:
            self.input_tokens += len(self.tokenizer.encode(prompt))

    def on_llm_new_token(self, token: str, **kwargs):
        self.output_tokens += 1

    def on_retriever_end(self, documents, **kwargs):
        for doc in documents:
            self.retrieved_doc_tokens += len(self.tokenizer.encode(doc.page_content))

    def get_usage(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retrieved_doc_tokens": self.retrieved_doc_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }