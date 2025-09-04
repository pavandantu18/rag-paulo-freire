from rag import RAG

if __name__ == "__main__":
    rag_system = RAG()

    rag_system.seed_vector_database(filepaths=[
        "./documents/paulo-freire-educacao-como-pratica-da-liberdade.txt",
        "./documents/paulo-freire-pedagogia-da-autonomia.txt",
        "./documents/paulo-freire-pedagogia-da-esperanca.txt",
        "./documents/paulo-freire-pedagogia-do-oprimido.txt",
        "./documents/paulo-freire-por-uma-pedagogia-da-pergunta.txt",
        "./documents/computacao-critica-capitulo-8-computadores.txt",
        "./documents/computacao-critica-capitulo-10-linguagens-de-programacao.txt"
    ])

    rag_system.launch_gradio()