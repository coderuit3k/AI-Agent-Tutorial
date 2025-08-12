from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_text_tool, image_tool, save_image_from_url, show_images
import os

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    output_type: str = Field(description="The type of result: 'text' or 'image'")
    summary: str
    sources: list[str]
    tools_used: list[str]
    image_urls: list[str] = []

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant. You may return either text or image-based results depending on the query.
            If the result is image-based (e.g., infographic, artwork, photo), return output_type as "image" and include image_urls.
            Otherwise, return output_type as "text".\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_text_tool, image_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("🤖 Please input your query 💬: ")
k = int(input("Please give the number of images you want to generate: "))
raw_response = agent_executor.invoke({"query": query})

try:
    response = parser.parse(raw_response.get("output"))

    if response.output_type == "text":
        print("📄 Summary:", response.summary)
    elif response.output_type == "image":
        print("🖼 Image URLs:")
        for url in response.image_urls:
            print(" -", url)
            save_image_from_url(url)
        show_images(query, response.image_urls)
    else:
        print("⚠️ Unknown output type")

except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)