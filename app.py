import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



from HTML_templates import css, bot_template, user_template



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def get_conv_chain(vectorstore, chat_history):
    llm = ChatOllama(
        model="mistral",
        temperature=0
    )

    retriever = vectorstore.as_retriever()

    conversation_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: "\n".join(
                f"{m.type}: {m.content}" for m in chat_history.messages
            ),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return conversation_chain

  

prompt = ChatPromptTemplate.from_template(
    """
you are a helpful assistant 
ansewr with only the given context
if answer is not in context, say 'I don't know'

chat history:
{chat_history}

Context:
{context}

question:
{question}
"""
)

def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDF Chat", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

    st.header("Chat with Multiple PDFs :books:")
    user_question = st.text_input("Ask a question about document...")

    if user_question and st.session_state.conversation:
        answer = st.session_state.conversation.invoke(user_question)

        st.session_state.chat_history.add_message(
            HumanMessage(content=user_question)
        )
        st.session_state.chat_history.add_message(
            AIMessage(content=answer)
        )

        st.write(
            bot_template.replace("{{MSG}}", answer),
            unsafe_allow_html=True
        )


    with st.sidebar:
        st.subheader("Your Document")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'.", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                if not pdf_docs:
                    st.warning("Please upload at least one PDF.")
                    return
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vector store 
                vectorstore = get_vector_store(text_chunks)

                # create conversation chain 
                st.session_state.conversation = get_conv_chain(
                    vectorstore,
                    st.session_state.chat_history
                )
    # st.session_state.conversation



if __name__ == '__main__':
    main()