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
        return PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    
    def _get_dcitionary_chain(self):
        dictionary = ["사람과 관련된 표현 -> 거주자"]
        dictionary_prompt = ChatPromptTemplate.from_template(f"""
            사용자의 질문을 보고, 우리의 사전을 참고 해서 사용자의 질문을 변경해주세요.
            만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 되고, 질문을 그대로 반환해주세요.
            사전: {dictionary}
            
            질문: {{query}}
        """)

        dictionary_chain = dictionary_prompt | self.llm | StrOutputParser()

        return dictionary_chain
    
    
    def _get_retriever(self):        
        return self.database.as_retriever(search_kwargs={"k": 3})
    
    
    def _get_qa_chain(self):
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self._get_retriever(),
            chain_type_kwargs={"prompt":self.prompt}
        )

        return qa_chain

    
    def get_ai_message(self, user_message):
        dictionary_chain = self._get_dcitionary_chain()
        qa_chain = self._get_qa_chain()        
        tax_chain = {"query": dictionary_chain} | qa_chain
        ai_message = tax_chain.invoke({"query": user_message})
        print(ai_message)

        return ai_message['result']