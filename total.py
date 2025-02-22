from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

# 모델 불러오기
llm = ChatOllama(model='llama3.2:1b')

# print(llm.invoke('what is the capital of China?'))

output_parser = StrOutputParser()
print(output_parser.invoke(llm.invoke('what is the capital of China?')))

# chain 1 - 나라를 제시하면 그 나라에서 유명한 음식을 답하기
## 프롬프트
prompt_template_food = PromptTemplate(
    template="What is the most famous food in {country}?, response only food name",
    input_variables=["country"],
)

prompt_template_recipe = PromptTemplate(
    template="Tell me about {food} recipe",
    input_variables=["food"],
)

food_chain = prompt_template_food | llm | output_parser
# print(food_chain.invoke({"country":"Italy"}))

recipe_chain = prompt_template_recipe | llm | output_parser
# print(recipe_chain.invoke({"food":"pizza"}))


total_chain = {"country":RunnablePassthrough()} | {"food":food_chain} | recipe_chain
print(total_chain.invoke('korea'))