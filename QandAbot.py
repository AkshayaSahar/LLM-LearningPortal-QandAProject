from langchain.llms import GooglePalm
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
load_dotenv()

llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0)


model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(
    model_name=model_name)

vector_db_fpath = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt')
    doc = loader.load()
    vector_db = FAISS.from_documents(documents=doc, embedding=hf)
    vector_db.save_local(vector_db_fpath)

#st.title("E-learning Q&A")

def get_qa_chain():
    # load vector db from local folder
    vectorDB = FAISS.load_local(vector_db_fpath, hf)
    retriever = vectorDB.as_retriever(score_threshold=0.7) # takes the input question and compares its embedding with vector db 
    prompt_template = """
    Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
    CONTEXT: {context}
    QUESTION: {question}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # forming the prompt template
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",    
                                        retriever=retriever, 
                                        input_key="query", 
                                        return_source_documents=True, 
                                        chain_type_kwargs={"prompt":PROMPT})
    return chain


if __name__ == "__main__":
    # create_vector_db()
    chain = get_qa_chain()
    print(chain("do you provide Internship?"))
