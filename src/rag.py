# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser


# def create_rag_chain(retriever):

#     llm = ChatOpenAI(
#         model="gpt-5.6-luna",
#         temperature=0.3,
#         max_tokens=512
#     )

#     template = """
# Answer the question based only on the following context.

# Context:
# {context}

# Question:
# {question}

# If the answer is not available in the context, say:
# Insufficient information in the uploaded documents.
# """

#     prompt = ChatPromptTemplate.from_template(template)

#     def format_docs(docs):
#         return "\n\n".join(
#             doc.page_content
#             for doc in docs
#         )

#     rag_chain = (
#         {
#             "context": retriever | format_docs,
#             "question": RunnablePassthrough()
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return rag_chain