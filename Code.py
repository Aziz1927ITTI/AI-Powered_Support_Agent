
# Salla RAG-Style Chatbot using OpenAI
# Requires: openai, pandas, numpy, sklearn

import openai
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ========== 1. Load OpenAI API Key ==========
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your key or use os.getenv() for security

# ========== 2. Load Dataset ==========
with open("salla_chatbot_qa_dataset_updated.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# ========== 3. Generate Embeddings for Questions ==========
# Use OpenAI Embedding API to convert questions into vector format
# We'll use 'text-embedding-3-small' (latest lightweight model)

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Apply embedding to all questions (only run once and save if needed)
print("Generating embeddings for all dataset questions...")
df["embedding"] = df["question"].apply(lambda q: get_embedding(q.lower()))

# ========== 4. Function to Find Similar Questions ==========
def find_similar_questions(user_question, top_n=3):
    user_embedding = get_embedding(user_question.lower())
    all_embeddings = np.vstack(df["embedding"].to_numpy())
    similarities = cosine_similarity([user_embedding], all_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    return df.iloc[top_indices], similarities[top_indices]

# ========== 5. Prompt Construction ==========
def build_prompt(user_input, matched_df):
    context = ""
    for idx, row in matched_df.iterrows():
        context += f"Q: {row['question']}\nA: {row['answer']}\n\n"
    prompt = f"""You are a helpful assistant for the eCommerce platform Salla.com.
Use only the information below to answer the user's question.

{context}

User: {user_input}
Answer:"""
    return prompt

# ========== 6. Chat with GPT ==========
def chat_with_gpt(user_input):
    matched_qs, scores = find_similar_questions(user_input)
    prompt = build_prompt(user_input, matched_qs)
    
    response = openai.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()

# ========== 7. Interactive Chat ==========
print("Salla GPT Chatbot is ready. Type 'exit' to quit.")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break
    try:
        reply = chat_with_gpt(user_input)
        print(f"SallaBot: {reply}")
    except Exception as e:
        print("Error:", e)
