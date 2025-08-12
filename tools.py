import openai
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import requests
import os
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def save_to_txt(data: str, file_name: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    try:
        with open(file_name, "a", encoding="utf-8") as f:
            f.write(formatted_text)
        print(f"Data successfully saved to {file_name}")
    except Exception as e:
        raise ValueError(f"Failed to save text: {e}")

def save_image_from_url(url: str, folder="images"):
    try:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        img_data = requests.get(url).content
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
        path = os.path.join(folder, file_name)

        with open(path, "wb") as f:
            f.write(img_data)

        print(f"Saved image to {path}")
    except Exception as e:
        raise ValueError(f"Failed to save image: {e}")

def generate_image_from_prompt(query: str) -> str:
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.images.generate(
            prompt=query,
            n=1,
            size="256x256"
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        return f"Image generation failed: {e}"

def show_images(query: str, urls: list[str]) -> None:
    if not urls:
        print("⚠️ No image URLs to display.")
        return

    plt.figure(figsize=(20, 5))

    plt.subplot(1, len(urls) + 1, 1)
    plt.text(0.5, 0.5, f"Query: {query}", fontsize=16, ha='center', va='center')
    plt.axis("off")
    start_idx = 2

    for i, url in enumerate(urls):
        plt.subplot(1, len(urls) + 1, start_idx + i)
        image = Image.open(BytesIO(requests.get(url).content))
        plt.title(f"Match {i + 1}:")
        plt.imshow(image)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

image_tool = Tool(
    name="generate_image",
    func=generate_image_from_prompt,
    description="Generate an image from a descriptive prompt. Input should be a detailed image description.",
)

save_text_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()

def search_limited(query: str, num_results: int) -> str:
    raw_results = search.run(query)
    results_list = raw_results.split("\n")
    return "\n".join(results_list[:num_results])

search_tool = Tool(
    name="search",
    func=lambda q: search_limited(q, num_results=1),
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)