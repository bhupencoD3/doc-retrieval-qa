from typing import List, Dict
from src.state.rag_state import RAGState
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langgraph.prebuilt import create_react_agent

class RAGNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None
        self._wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en"))

    def _build_tools(self) -> List[Tool]:
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge",
            func=self._wiki_tool.run
        )
        return [wikipedia_tool]

    def _build_agent(self):
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. Prefer user-provided documents first; "
            "use 'wikipedia' only if documents are insufficient. Return only the final useful answer."
        )
        self._agent = create_react_agent(self.llm, tools=tools, state_modifier=system_prompt)

    def generate_answer(self, state: RAGState) -> Dict:
        question = state.get("question", "")
        # Step 1: Retrieve docs
        retrieved_docs = self.retriever.invoke(question)

        # Step 2: Prepare content for summarization
        if retrieved_docs:
            # Combine document content (up to 3 documents) for summarization
            context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])
            # Create a prompt to summarize the retrieved content
            prompt = [
                SystemMessage(content=(
                    "You are a helpful assistant. Summarize the following information "
                    "in 2-3 concise sentences to directly answer the user's question. "
                    "Focus on the main ideas and avoid including unnecessary details."
                )),
                HumanMessage(content=f"Question: {question}\n\nContext: {context}")
            ]
            # Generate summarized answer using the LLM
            answer = self.llm.invoke(prompt).content.strip()
        else:
            # Step 3: No docs → fallback to ReAct (Wikipedia)
            if self._agent is None:
                self._build_agent()
            result = self._agent.invoke({"messages": [HumanMessage(content=question)]})

            from langchain_core.messages import AIMessage, ToolMessage
            answer_parts = []

            messages = []
            if isinstance(result, dict) and "messages" in result:
                messages = result["messages"]
            elif isinstance(result, list):
                messages = result
            elif isinstance(result, AIMessage) or isinstance(result, ToolMessage):
                messages = [result]

            # Collect content from Wikipedia results
            for msg in messages:
                content = getattr(msg, "content", None)
                if content:
                    answer_parts.append(content)

                tool_calls = getattr(msg, "tool_calls", [])
                for call in tool_calls:
                    for key in ["result", "output", "output_text"]:
                        if call.get(key):
                            answer_parts.append(call[key])

            combined_content = "\n\n".join(answer_parts) if answer_parts else "Could not generate the answer"
            
            # Summarize Wikipedia content if it's too long
            if len(combined_content) > 500:  # Arbitrary threshold to trigger summarization
                prompt = [
                    SystemMessage(content=(
                        "You are a helpful assistant. Summarize the following information "
                        "in 2-3 concise sentences to directly answer the user's question. "
                        "Focus on the main ideas and avoid including unnecessary details."
                    )),
                    HumanMessage(content=f"Question: {question}\n\nContext: {combined_content}")
                ]
                answer = self.llm.invoke(prompt).content.strip()
            else:
                answer = combined_content

        # Return a dict as required by LangGraph
        return {"answer": answer, "retrieved_docs": retrieved_docs}