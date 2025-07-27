import pandas as pd
from langchain.vectorstores import Chroma
import tempfile
...
persist_directory = tempfile.mkdtemp()
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class RAGChatbot:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.docs = self._generate_docs()
        self.vectorstore = self._build_vectorstore()
        self.qa_chain = self._build_qa_chain()

    def _generate_docs(self):
        docs = []
        for _, row in self.df.iterrows():
            text = f"Applicant {row['Loan_ID']} is a {row['Gender']} with {row['Education']} education, "
            text += f"has an income of {row['ApplicantIncome']} and applied for a loan of {row['LoanAmount']}. "
            text += f"Loan status: {row['Loan_Status']}."
            docs.append(Document(page_content=text))
        return docs

    def _build_vectorstore(self):
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(self.docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        persist_directory = tempfile.mkdtemp()
        vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
        return vectorstore

    def _build_qa_chain(self):
        # Use HuggingFace transformers to create a QA pipeline
        model_name = "google/flan-t5-base"  # Free and lightweight
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        return qa_chain

    def get_answer(self, query):
        return self.qa_chain.run(query)
