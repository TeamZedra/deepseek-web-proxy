from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1337/v1",
    api_key="nah",
)

response = client.chat.completions.create(
    model="deekseek-chat",
    messages=[{"role": "user", "content": "Hello DeepSeek, I am using a wonderful web proxy by TeamZedra to chat with you!"}],
    stream=True,
)

for chunk in response:
    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
    if reasoning:
        print(reasoning, end="", flush=True)
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)