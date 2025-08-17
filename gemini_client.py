import google.generativeai as genai

class GeminiClient:
    def __init__(self, api_key: str, gen_model="gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.gen_model = genai.GenerativeModel(gen_model)



    def answer(self, prompt: str, temperature=0.8, max_output_tokens=1024) -> str:
        resp = self.gen_model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens}
        )
        # print(resp.text)
        return resp.text
