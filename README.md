🤖 AI-Powered Support Agent for Salla.com
This project presents an intelligent chatbot solution designed to automate customer support for the Salla.com e-commerce platform. The system combines semantic search with OpenAI's GPT-4 to understand and respond to customer queries in a natural and accurate way.

🚀 Features
Retrieval-Augmented Generation (RAG) architecture

Uses OpenAI text-embedding-3-llama for semantic search

GPT-4 powered response generation

Supports various formats of customer questions

Modular Python implementation (ready for web/mobile expansion)

🧠 How It Works
A customer sends a question.

The question is embedded into a vector using OpenAI embeddings.

The system retrieves the top 3 most similar questions using cosine similarity.

A prompt is built using those Q&A pairs and sent to GPT-4.

GPT-4 generates and returns the final answer.

📊 Model Evaluation
We evaluated the system using 20 test questions (both seen and unseen), and achieved:

Top-3 Accuracy: 90%

Mean Reciprocal Rank (MRR): 0.78

Exact Match: 70%

Semantic Correctness: 80%

Robustness: 100% (on vague or irrelevant questions)

⚙️ Tech Stack
Python

OpenAI API (Embeddings + GPT-4)

NumPy & Pandas

Cosine similarity

PyCharm IDE

📌 Challenges
OpenAI API usage costs

Handling vague/short questions

Not yet integrated with Salla's live backend

✅ Conclusion
This project demonstrates how combining semantic retrieval with GPT-4 can significantly improve customer support efficiency and scalability in real-world e-commerce platforms like Salla.

