from openai import OpenAI
import httpx

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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",
  http_client=httpx.Client(verify=False)    # <-- only for local/testing
)

completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[{"role":"user","content":"provide me an article on agentic rag"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=False
)

# print(completion.choices[0].message)
print(completion)

