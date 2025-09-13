from google import genai

client = genai.Client(api_key="AIzaSyDlqO6jC-74q8IKmpicT20SEA8SSpNiB6c")

response = client.models.generate_content(
    model="gemini-1.5-flash-8b",
    contents="what date is it today",
    config={
        "temperature": 0,
        "max_output_tokens": 500  # 设置最大输出 token 数
    }
)
print(response.text)
