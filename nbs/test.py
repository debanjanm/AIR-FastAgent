from ollama import chat

# from pathlib import Path

# Pass in the path to the image
path = input("test.jpg")

# You can also pass in base64 encoded image data
# img = base64.b64encode(Path(path).read_bytes()).decode()
# or the raw bytes
# img = Path(path).read_bytes()

response = chat(
    model="qwen2.5vl",
    messages=[
        {
            "role": "user",
            "content": "What is in this image? Be concise.",
            "images": ["test.jpg"],
        }
    ],
)

print(response.message.content)
