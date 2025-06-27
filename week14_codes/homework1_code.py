"""
通过langchain实现特定主题聊天系统，支持多轮对话。
"""

from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI # LLM调用封装
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # 对话角色：user、assistant、system
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
import json
from datetime import datetime

# 1. 导入必要包
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 导入template中传递聊天历史信息的“占位”类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def save_history(session_id):
    # Message对象中to_json()消息转换json格式
    history = [msg.to_json() for msg in store[session_id].messages]
    with open(f"history_{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 实现：聊天历史保存（不同session_id）
def load_history(session_id):
    # 加载失败情况下，跳转到except（错误）
    try:
        with open(f"history_{session_id}.json", "r", encoding="utf-8") as f:
            messages = json.load(f)
        store[session_id] = ChatMessageHistory()
        store[session_id].parse_json(messages)
    except FileNotFoundError:
        store[session_id] = ChatMessageHistory()


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # Initialize the OpenAI Chat model
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )
 
    # 带有占位符的prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个得力的助手。你可以使用{lang}回答用户的问题。"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    parser = StrOutputParser()
    # chain 构建
    chain =  prompt | model | parser
    
    # 定制存储消息的dict
    store = {}

    # 定义函数：根据sessionId获取聊天历史（callback 回调）    
    def get_session_history(session_id):
        if session_id not in store:
            load_history(session_id)
        return store[session_id]

    # 在chain中注入聊天历史消息
    with_msg_hist = RunnableWithMessageHistory(
        chain, 
        get_session_history=get_session_history,
        input_messages_key="messages")
    
    # 动态输入session_id
    session_id = input('请输入session_id（用于区分不同会话）：').strip()

    while True:
        user_input = input('用户输入的Message (输入 "quit" 退出)：')
        if user_input.lower() == 'quit':
            break

        # 调用注入聊天历史的对象
        response = with_msg_hist.invoke(
            {
                "messages":[HumanMessage(content=user_input)],
                "lang":"汉语"
            },
            config={'configurable':{'session_id': session_id}})

        print('AI Message:', response)
        # 聊天历史自动保存
        save_history(session_id)