# 🤖 Intelligent Agent for Women's Football EuroCup 2025

This project is a conversational agent built in Python using LangChain. It’s designed to answer questions about women’s football euro cup using relevant and curated sources. It leverages a Retrieval-Augmented Generation (RAG) ans SQL queries to a PostgreSQL Database using custom tools.

---

## 🎯 Goal

To improve access to information about women’s football — including countries, stadiums, players, fixture and statistics — **which is often scattered or underrepresented in mainstream sources**.

---

## 🛠️ Tech Stack

- **Python 3.12**
- **LangChain**
- **OpenAI API**
- **RAG** for document-based retrieval
- **SQL** Tool for structured queries
- **Pinecone** for vector search
- **PostgreSQL** as relational data source
- Custom embeddings
- Structured logging, modular architecture, and support for future extensions

---

## 📁 Project Structure

- `agents/`: conversational agent logic
- `rag/`: retrieval logic, embeddings, and vector store configs
- `tools/`: specialized tools like SQL or competition rules
- `services/`, `dto/`, `config/`, `utils/`: clean and scalable architecture
- `notebooks/`: for testing and data loading

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/AleGallagher/UERO2025-Agent.git
   cd UERO2025-Agent

2. Run the app with Docker:
   ```bash
   docker-compose up --build

3. Make sure you create a .env file with the required API keys

## 🌐 Live Demo
You can try the agent through the deployed front-end here:

👉 https://weuro2025agent.vercel.app/ ← (replace with your actual URL)


## ⚠️ Note


This project is still in early development and will be improved incrementally.
Feedback, ideas, and contributions are very welcome!


## 📌 Roadmap (Next Steps)

* 🧠 Migration to LangGraph for more robust control flow

* 🌍 Expand document base to include more tournaments and sources

* 🔄 Enable **streaming responses** for faster and more natural interaction  


## Authors

- [@AleSandler](https://github.com/AleGallagher)


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/AleGallagher)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alejandro-sandler-66ba4254/)