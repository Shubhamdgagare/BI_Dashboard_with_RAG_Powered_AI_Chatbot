# app.py
import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from ingestion.verify_store import get_metadata_table

# --- Initialization ---
env_path = os.path.join(os.path.dirname(__file__), "config", ".env")
load_dotenv(dotenv_path=env_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set.")

SQLITE_DB = "db/sqlite/structured.db"
VECTOR_DB_DIR = "db/vectorstore"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GEMINI_API_KEY, temperature=0.0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings, collection_name="unstructured_data")

metadata_cache = get_metadata_table(SQLITE_DB)

app = Flask(__name__)
CORS(app)

# --- Helper Functions ---

def query_router(user_input: str) -> str:
    table_summaries = "\n".join([f"- {name}: Contains columns like {', '.join(meta['columns'])}" for name, meta in metadata_cache.items()])
    prompt = (
        f"You are an expert query routing system. Classify the user's request into one of the following categories:\n\n"
        f"1. 'GREETING': For simple greetings.\n"
        f"2. 'SQL_DATABASE': For specific questions that refer to data in the database tables OR if the user explicitly mentions the 'dashboard'.\n"
        f"3. 'DOCUMENT_SEARCH': For definitional questions ('what is...') or questions about proper nouns not in the database (e.g., 'shubham').\n"
        f"4. 'AMBIGUOUS_SQL': For broad questions about data that don't specify a source table (e.g., 'list companies').\n\n"
        f"Database Tables:\n{table_summaries}\n\n"
        f"User's request: \"{user_input}\"\n\n"
        f"Return only the category name: GREETING, DOCUMENT_SEARCH, SQL_DATABASE, or AMBIGUOUS_SQL."
    )
    response = llm.invoke(prompt)
    return response.content.strip()

def generate_sql(user_input: str, is_ambiguous: bool = False) -> str:
    """Generates and robustly cleans an SQL query from the LLM's response."""
    all_schemas = "\n".join([f"- Table: {name}\n  Columns: {meta['columns']}" for name, meta in metadata_cache.items()])
    ambiguity_instruction = ""
    if is_ambiguous:
        ambiguity_instruction = "The user's query is ambiguous. Assume they are asking about the 'aviso_technographics' table."

    prompt = (
        f"You are an expert SQLite analyst. You have access to these tables:\n\n{all_schemas}\n\n"
        f"{ambiguity_instruction}\n\n"
        f"Based on the user's question, write a single, valid SQLite query to answer it. You can JOIN tables on common columns.\n"
        f"IMPORTANT: Before finishing, double-check that the column names in your query EXACTLY MATCH the column names in the schemas provided.\n\n"
        f"User Question: '{user_input}'\n\n"
        f"Return only the SQLite query."
    )
    response = llm.invoke(prompt)
    raw_sql = response.content.strip().replace("```sql", "").replace("```", "")

    select_pos = raw_sql.upper().find("SELECT")
    if select_pos != -1:
        cleaned_sql = raw_sql[select_pos:]
        return cleaned_sql.strip()
    else:
        return raw_sql.strip()

def make_unique_if_single_column(sql: str) -> str:
    """
    If the query selects only a single column and doesn't already use DISTINCT, 
    modify it to SELECT DISTINCT.
    """
    sql_stripped = sql.strip()
    sql_lower = sql_stripped.lower()

    if sql_lower.startswith("select") and " from " in sql_lower:
        # Extract the part after SELECT and before FROM
        select_part = sql_stripped.split("FROM", 1)[0]
        # Remove SELECT keyword and split by comma
        columns = [col.strip() for col in select_part[6:].split(",")]

        # Only apply if there’s exactly one column and DISTINCT is not already present
        if len(columns) == 1 and "distinct" not in select_part.lower():
            sql_stripped = sql_stripped.replace("SELECT", "SELECT DISTINCT", 1)

    return sql_stripped

def run_sql(sql: str) -> dict:
    """Runs the SQL query and formats the output."""
    try:
        with sqlite3.connect(SQLITE_DB) as conn:
            df = pd.read_sql(sql, conn)
        if df.empty:
            return {"type": "text", "data": "No data was found for your query."}
        return {"type": "table", "data": df.to_html(index=False, classes='sql-result-table'), "raw_data": df.to_dict(orient='records')}
    except Exception as e:
        print(f"SQL Error: {e}\nFailed Query: {sql}")
        return {"type": "text", "data": "I encountered an error trying to query the database. The data might not be available or the query may be incorrect."}

def search_documents(user_input: str) -> dict:
    """Searches documents and generates a response."""
    docs = vectorstore.similarity_search(query=user_input, k=3)
    if not docs:
        return {"type": "text", "data": "I could not find any relevant information in the documents."}
    
    context = "\n\n".join([d.page_content for d in docs])
    final_prompt = (
        f"Answer the user's question based ONLY on the following context.\n\n"
        f"Context:\n{context}\n\nQuestion: {user_input}"
    )
    response = llm.invoke(final_prompt)
    return {"type": "text", "data": response.content.strip()}

# --- API Endpoint ---
@app.route("/chat", methods=["POST"])
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message")
        if not user_message:
            return jsonify({"error": "No message provided."}), 400

        route = query_router(user_message)
        print(f"DEBUG: Routed query to: {route}")

        if route == "GREETING":
            bot_response = {"type": "text", "data": "Hello! How can I assist you with your data today?"}

        elif route == "SQL_DATABASE":
            sql_query = generate_sql(user_message)
            sql_query = make_unique_if_single_column(sql_query)
            bot_response = run_sql(sql_query)
            bot_response["sql_query"] = sql_query

        elif route == "AMBIGUOUS_SQL":
            sql_query = generate_sql(user_message, is_ambiguous=True)
            sql_query = make_unique_if_single_column(sql_query)
            bot_response = run_sql(sql_query)
            bot_response["sql_query"] = sql_query

        elif route == "DOCUMENT_SEARCH":
            bot_response = search_documents(user_message)

        else:
            bot_response = search_documents(user_message)

        return jsonify(bot_response)

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"type": "text", "data": "An error occurred on the server."}), 500


@app.route("/status")
def status():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5001)
