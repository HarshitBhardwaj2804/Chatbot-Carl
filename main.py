from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils import NamedJSONChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from extensions.search_engine import TavilyAgent, TavilyAgentwithHistory
from datetime import datetime
import random, string
import json

#0 #Setting up the enviroment
load_dotenv()

## Setting-Up Langchain-tracing.
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CARL"

## Setting up the GROQ API KEY
groq_api_key = os.getenv("GROQ_API_KEY_MAIN_PROJECT")

prompt = ChatPromptTemplate(
    [
        ("system", ("You are CARL, an excellent assistant. Your task is to help your master as best you can in his/her questions. \n"
        "You have to answer in a way define in the custom instruction given by the user. If the value of instructions \n"
        "is None/Null then you just have to answer in a normal manner . \n"
        "Try to answer not very long it loose attention, answer long when there is a need."
        "{instructions}")),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{question}")
    ]
)

# llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)
# chain = prompt | llm

def get_session_history(session_id: str):
    return NamedJSONChatMessageHistory(session_id=session_id, file_path="chat_sessions_test.json")

# message_chain = RunnableWithMessageHistory(
#     chain,
#     get_session_history=get_session_history,
#     input_messages_key="question",
#     history_messages_key="history"
# )

def generate_session_details():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    rand_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    session_id = f"session_{timestamp}-{rand_suffix}"
    return session_id

SETTINGS_FILE = "user_settings.json"

def get_settings():
    if not os.path.exists(SETTINGS_FILE):
        return {"model": "openai/gpt-oss-120b", "temperature": 0.7, "instructions": ""}
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)

app = Flask(__name__)

@app.route("/")
def index():
    session_id = generate_session_details()
    settings = get_settings()
    is_dark_mode = settings.get("darkMode", False)
    return render_template("index.html", session_id=session_id, dark_mode=is_dark_mode)
    

@app.route("/send_message", methods=["POST"])
def send_message():
    user_message = request.json.get("message")
    webSearchFlag = request.json.get("webSearchFlag")
    session_id = request.json.get("session_id")

    settings = get_settings()
    instructions = settings.get("instructions")
    model_name = settings.get("model")
    temperature = settings.get("temperature")

    # Use the new values to instantiate the LLM and the chain dynamically
    llm = ChatGroq(model=model_name, groq_api_key=groq_api_key, temperature=temperature)
    chain = prompt | llm

    message_chain = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    if webSearchFlag == 1:
        search_agent = TavilyAgentwithHistory()
        
        # response = search_agent.run(user_message)
        agent_with_history = RunnableWithMessageHistory(
            search_agent.executor,
            get_session_history=get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )
        response = agent_with_history.invoke(
            {"question":user_message},
            config={"configurable":{"session_id":session_id}}
        )
        return jsonify({"reply": response["output"]})
    else:
        response = message_chain.invoke(
        {"question": user_message,
         "instructions" : instructions},
        config={"configurable": {"session_id": session_id}}
        )
        
        return jsonify({"reply": response.content})

@app.route("/history")
def history():
    file_path = "chat_sessions_test.json"
    if not os.path.exists(file_path):
        return render_template("history.html", sessions=[], last_session=None)

    with open(file_path, "r") as f:
        all_data = json.load(f)

    last_session_id = all_data.get("last_session_id")
    sessions = []
    last_session = None

    for session_id, messages in all_data.items():
        if session_id == "last_session_id":
            continue

        if not messages or "chat_name" not in messages[0]:
            continue

        chat_name = messages[0]["chat_name"].strip()
        first_user_msg = next((m["content"] for m in messages if m.get("type") == "human"), "N/A")

        # Extract timestamp
        parts = session_id.partition("_")[2].partition("-")[0]
        datestamp_str = parts[:8]
        timestamp_str = parts[8:12]

        try:
            date_obj = datetime.strptime(datestamp_str, "%Y%m%d")
            formatted_date = date_obj.strftime("%B %d, %Y")
            time_obj = datetime.strptime(timestamp_str, "%H%M")
            formatted_time = time_obj.strftime("%I:%M %p")
        except Exception:
            formatted_date = "Unknown"
            formatted_time = ""

        date_time = f"{formatted_date} : {formatted_time}"
        session_obj = {
            "session_id": session_id,
            "chat_name": chat_name,
            "first_message": first_user_msg,
            "datetime": date_time
        }

        sessions.append(session_obj)

        if session_id == last_session_id:
            last_session = session_obj

    sessions.reverse()

    return render_template("history.html", sessions=sessions, last_session=last_session)


@app.route("/load_chat/<session_id>")
def load_chat(session_id):
    return render_template("index.html", session_id=session_id)

@app.route("/get_previous_messages/<session_id>")
def get_previous_messages(session_id):
    file_path = "chat_sessions_test.json"
    if not os.path.exists(file_path):
        return jsonify([])

    with open(file_path, "r") as f:
        all_data = json.load(f)

    messages = all_data.get(session_id, [])
    return jsonify(messages)

@app.route("/delete_chat/<session_id>", methods=["DELETE"])
def delete_chat(session_id):
    file_path = "chat_sessions_test.json"
    if not os.path.exists(file_path):
        return jsonify({"success": False, "error": "File not found"}), 404

    try:
        with open(file_path, "r") as f:
            all_data = json.load(f)

        if session_id not in all_data:
            return jsonify({"success": False, "error": "Session not found"}), 404

        # Remove the session
        del all_data[session_id]

        with open(file_path, "w") as f:
            json.dump(all_data, f, indent=2)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/extensions")
def extensions():
    return render_template("extensions.html")

## Extension-1: WEB SEARCH
@app.route("/search_engine")
def search_engine():
    return render_template("search_engine.html")

@app.route("/webSearch", methods=["POST"])
def webSearch():
    user_message = request.json.get("message")

    web_agent = TavilyAgent()
    executor = web_agent.executor
    response = executor.invoke({"question" : user_message})

    return jsonify({"reply" : response["output"]})

## EXTENSION-2: RAG PDF CHATBOT
@app.route("/PDF_Chatbot")
def pdf_chatbot():
    return render_template("pdf_chatbot.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")

@app.route("/get_settings")
def get_settings_route():
    settings = get_settings()
    return jsonify(settings)

@app.route("/save_settings", methods=["POST"])
def save_settings():
    try:
        settings_data = request.json
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings_data, f, indent=2)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
