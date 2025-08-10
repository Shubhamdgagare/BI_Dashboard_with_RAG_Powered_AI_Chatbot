# BI_Dashboard_with_RAG_Powered_AI_Chatbot

YouTube Video - https://youtu.be/rbg9KOBJUnc


# BI Dashboard Chatbot

A conversational AI assistant for querying both structured (SQL) and unstructured (documents) business data.  
Built with Flask, LangChain, ChromaDB, and Gemini (Google Generative AI).

---

## Features

- **Chatbot API**: Natural language interface for business data.
- **Structured Data**: Query CSV/XLSX files ingested into SQLite.
- **Unstructured Data**: Search and summarize PDFs, DOCX, and TXT files via vector embeddings.
- **Semantic Routing**: Automatically routes queries to SQL or document search.
- **Metadata Awareness**: Uses table/file metadata for context.

---

## Folder Structure

```
BI_Dashboard_Chatbot/
│
├── Chatbot_Server.py         # Flask API server
├── ingestion/
│   ├── Data_Ingestion.py     # Data ingestion script
│   └── verify_store.py       # Data verification utility
├── db/
│   ├── sqlite/               # SQLite DB location
│   └── vectorstore/          # ChromaDB vector store
├── data/
│   ├── structured/           # Place CSV/XLSX files here
│   └── unstructured/         # Place PDF/DOCX/TXT files here
├── config/
│   └── .env                  # API keys and config
├── requirements.txt
└── README.md
```

---

## Setup Instructions

1. **Clone the repository**

    ```sh
    git clone https://github.com/yourusername/BI_Dashboard_Chatbot.git
    cd BI_Dashboard_Chatbot
    ```

2. **Install dependencies**

    ```sh
    pip install -r requirements.txt
    ```

3. **Configure environment variables**

    - Create a `.env` file in the `config/` folder:
      ```
      GEMINI_API_KEY=your_google_gemini_api_key
      ```

4. **Add your data**

    - Place structured files (`.csv`, `.xlsx`) in `data/structured/`
    - Place unstructured files (`.pdf`, `.docx`, `.txt`) in `data/unstructured/`

5. **Ingest data**

    ```sh
    python ingestion/Data_Ingestion.py
    ```

6. **Verify data (optional)**

    ```sh
    python ingestion/verify_store.py
    ```

7. **Run the chatbot server**

    ```sh
    python Chatbot_Server.py
    ```

    The API will be available at [http://localhost:5001](http://localhost:5001).

---

## API Usage

- **POST /chat**
    - Request: `{ "message": "Your question here" }`
    - Response: JSON with answer or table

- **GET /status**
    - Health check endpoint

---

## Notes

- Requires a valid [Google Gemini API key](https://ai.google.dev/).
- For best results, use Python 3.9+.

---

## License

MIT
>>>>>>> 2f3ae3e (Initial commit: BI Dashboard Chatbot)
