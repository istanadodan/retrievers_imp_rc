import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from template.htmlTemplates import css, bot_template, user_template
from utils import print_log
import logging

@print_log
def get_pdf_text(pdf_docs):
    raw_text = ''
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            raw_text += page.extract_text()
    return raw_text    

@print_log
def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = splitter.split_text(raw_text)
    return text_chunks    

@print_log
def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)    
    return vectorstore    

@print_log
def get_local_vectorstor(dir:str, file:str):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.load_local(dir, index_name=file, embeddings=embeddings)    
    return vectorstore    

@print_log
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                               retriever=vectorstore.as_retriever(),
                                                               memory=memory)
    return conversation_chain

@print_log
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})    
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 ==0 :
            st.write(user_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)
                              
def logger_setting():    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(funcName)s %(name)s:%(lineno)d %(message)s"
    )
                
def main():
    load_dotenv()        
    logger_setting()
    
    logging.info(f'started') 
    st.set_page_config(page_title='chat with multiple pdfs', page_icon=':books:')
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
        
    st.write(css, unsafe_allow_html=True)
    
    st.header('chat with multiple pdfs :books:')
    user_question = st.text_input('Ask question about your documents')     
    logging.info(f'user_question: {user_question}') 
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)    
        
    with st.sidebar:        
        st.header('Your Documents')
        pdf_docs = st.file_uploader('Upload your documents', type=['pdf'], accept_multiple_files=True)
        logging.info(f'pdf_docs: {pdf_docs}')
        db_dir = os.path.join(os.path.dirname(__file__), 'db')
        # pdf_docs에서 얻은 파일명목록을 이어서 파일명으로 사용한다.        
        db_file = f'{"_".join(list(map(lambda x: x.name.replace(".pdf",""), pdf_docs)))}'
        logging.info(f'db_file: {db_file}')
        
        if st.button('Search'):
            # vectorstore에 저장된 db를 가져온다.
            filepath = os.path.join(db_dir, f'{db_file}.faiss')
            logging.info(f'filepath: {filepath}')
            if os.path.exists(filepath):
                with st.spinner('Proccessing'):
                    vectorstore = get_local_vectorstor(db_dir, db_file)
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
            else:
                # 파일명이 없다는 팝업을 띄운다.
                st.write('<script>alert("Please upload your documents first")</script>', unsafe_allow_html=True)                
                return
        
        if st.button('Process'):
            with st.spinner('Proccessing'):
                # get pdf text 
                raw_text = get_pdf_text(pdf_docs)
                
                # get the text chunks
                text_chunk = get_text_chunks(raw_text)
                
                # create vector store
                vectorstore = get_vectorstore(text_chunk)
                vectorstore.save_local(db_dir, db_file)
    
                # create conversation chain
                st.write('completed')                

if __name__ == '__main__':    
    main()
