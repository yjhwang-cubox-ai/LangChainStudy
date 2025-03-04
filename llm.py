import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class AIBot:
    def __init__(self):
        self.database = self._read_database()
        self.llm = ChatOpenAI(model='gpt-4o-mini')
        self.prompt = hub.pull("rlm/rag-prompt")

    def _read_database(self):
        load_dotenv()
        embedding = OpenAIEmbeddings(model="text-embedding-3-large")
        index_name = 'test'
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        return PineconeVectorStore(index_name=index_name, embedding=embedding)
    
    def get_ai_message(self, query):

        dictionary = ["사람과 관련된 표현 -> 거주자"]
        dictionary_prompt = ChatPromptTemplate.from_template(f"""
            사용자의 질문을 보고, 우리의 사전을 참고 해서 사용자의 질문을 변경해주세요.
            만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 되고, 질문을 그대로 반환해주세요.
            사전: {dictionary}
            
            질문: {{query}}
        """)

        dictionary_chain = dictionary_prompt | self.llm | StrOutputParser()

        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.database.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt":self.prompt}
        )        

        tax_chain = {"query": dictionary_chain} | qa_chain

        # print(tax_chain.invoke({"query": query}))
        return tax_chain.invoke({"query": query})['result']