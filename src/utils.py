from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from typing import List
import os, json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY_MAIN_PROJECT")

class NamedJSONChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, file_path: str = "chat_sessions_test.json"):
        self.session_id = session_id
        self.file_path = file_path
        self.messages = self._load_messages()

    def _load_messages(self) -> List[BaseMessage]:
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r") as f:
            all_data = json.load(f)
        session_data = all_data.get(self.session_id, [])
        return [self._dict_to_message(m) for m in session_data if "type" in m]

    def _save_messages(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                try:
                    all_data = json.load(f)
                except json.JSONDecodeError:
                    all_data = {}
        else:
            all_data = {}

        # Update last_session_id
        all_data["last_session_id"] = self.session_id

        # Preserve chat_name if it exists
        session_data = all_data.get(self.session_id, [])
        chat_name_entry = next((item for item in session_data if "chat_name" in item), None)

        all_data[self.session_id] = []
        if chat_name_entry:
            all_data[self.session_id].append(chat_name_entry)

        all_data[self.session_id].extend([self._message_to_dict(m) for m in self.messages])

        with open(self.file_path, "w") as f:
            json.dump(all_data, f, indent=2)


    def _generate_chat_name(self, message: HumanMessage) -> str:
        instruction = """
        You are an intelligent AI assistant. Read the given message and generate a name (3 to 5 words max) for the conversation.
        Do NOT generate any code or explanation, just a short title based on the user's intent.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", instruction),
            ("user", "{question}")
        ])
        parser = StrOutputParser()
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
        name_chain = prompt | llm | parser
        return name_chain.invoke({"question": message.content}).strip()

    def add_message(self, message: BaseMessage) -> None:
        # If it's the first message and it's from the human, generate and store chat_name
        if not self.messages and isinstance(message, HumanMessage):
            chat_name = self._generate_chat_name(message)
            self._insert_chat_name(chat_name)

        self.messages.append(message)
        self._save_messages()

    def _insert_chat_name(self, name: str):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                all_data = json.load(f)
        else:
            all_data = {}

        all_data[self.session_id] = [{"chat_name": name}]
        with open(self.file_path, "w") as f:
            json.dump(all_data, f, indent=2)

    def clear(self) -> None:
        self.messages = []
        self._save_messages()

    def _message_to_dict(self, message: BaseMessage):
        return {"type": message.type, "content": message.content}

    def _dict_to_message(self, data: dict) -> BaseMessage:
        if data["type"] == "human":
            return HumanMessage(content=data["content"])
        elif data["type"] == "ai":
            return AIMessage(content=data["content"])
        else:
            raise ValueError("Unknown message type")
