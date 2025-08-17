import google.generativeai as genai
genai.configure(api_key="AIzaSyCfjmXl9uqibYW7PVouoJfGVUtvBtKrzFg")

resp = genai.embed_content(model="text-embedding-004", content="This is a test.")
print(type(resp))
