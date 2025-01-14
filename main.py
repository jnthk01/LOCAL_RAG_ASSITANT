import os
import asyncio
import hashlib
import tempfile
import streamlit as st
from langchain import hub
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()
GEMINI_KEY = os.getenv('GEMINI_API_KEY')

st.title("Local Assistant")
uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    files_names = [uploaded_file.name for uploaded_file in uploaded_files]
    sorted(files_names)
    file_names_combined = "_".join([uploaded_file.split('.')[0] for uploaded_file in files_names])
    PERSIST_DIRECTORY =os.path.join(os.getcwd(),f'chroma_db/{file_names_combined}') 

    if not os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            collection_name="embed-text",
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
            persist_directory=PERSIST_DIRECTORY,
        )

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            async def load_pages():
                loader = PyPDFLoader(temp_file_path)
                pages = [page async for page in loader.alazy_load()]
                return pages

            pages = asyncio.run(load_pages())
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
            chunks = text_splitter.split_documents(pages)

            for page_number, chunk in enumerate(chunks):
                metadata = {"file_name": file_name, "page_number": page_number + 1}
                vector_db.add_documents([chunk], metadata=[metadata])


    else:
        vector_db = Chroma(
            collection_name="embed-text",
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
            persist_directory=PERSIST_DIRECTORY,
        )

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=None,
        timeout=None
    )

    system_prompt = (
        "This is a RAG retriever which provides insights from the uploaded text using AI."
        " Your purpose is to give the user a simplified answer in an easily understandable and neatly formatted way."
        " There might be chat history provided, in the format of System Output: and User Input:"
        " Wherever you find the above format, consider it as chat history."
        "\n\n"
        "{context}"
    )

    human_prompt = (
        "This is the chat history till now for this conversation, if there is no history then don't consider it. Consider this also while giving your answer:"
        "{history}"
        "\n\n"
        "Now considering all the things above, give me the answer for this question:"
        "{input}"
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        runnable=retrieval_chain,
        rag_chain=retrieval_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello, this is your local assistant."})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your queries."):

        user_input = prompt

        with st.chat_message("user"):
            st.markdown(user_input)
            
        session_id = "user_session_1"
        chat_history_obj = get_session_history(session_id)
        hist = "\n".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"])

        result = conversational_rag_chain.invoke(
            {"input": user_input, "chat_history": hist},
            {"configurable": {"session_id": session_id}}
        )

        response = result["answer"]

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})