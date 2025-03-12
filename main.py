"""
title: DeepSeek R1
author: Novagrid.AI & zgccrui
description:  support Searxng and insert search results into the context (without specifically accessing pages), while displaying the chain of thought for the DeepSeek R1 model - only supported in version 0.5.6 and above. Fix the issue where DeepSeek R1 generated content lacks the opening <think> tag, address errors in web search and title generation, and achieve this by adding logic to avoid adding the <think> tag during non-streaming generation.
version: 1.2.10-r2
licence: MIT
"""

import json
import httpx
import re
from typing import AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field
import asyncio
import requests


class Pipe:
    class Valves(BaseModel):
        DEEPSEEK_API_BASE_URL: str = Field(
            default="https://api.deepseek.com/v1",
            description="DeepSeek API的基础请求地址",
        )
        DEEPSEEK_API_KEY: str = Field(
            default="", description="用于身份验证的DeepSeek API密钥，可从控制台获取"
        )
        DEEPSEEK_API_MODEL: str = Field(
            default="deepseek-reasoner",
            description="API请求的模型名称，默认为 deepseek-reasoner ",
        )
        DEEPSEEK_MODEL_DISPLAY_NAME: str = Field(
            default="deepseek-reasoner-model",
            description="模型名称，默认为 deepseek-reasoner-model",
        )
        searxng_url: str = Field(
            default="http://searxng:8080", description="SearXNG 的搜索服务地址"
        )
        enable_searxng: bool = Field(default=True, description="是否启用 SearXNG 搜索")
        searxng_results: int = Field(
            default=5, description="SearXNG 返回的最大结果数量"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.data_prefix = "data:"
        self.emitter = None

    def pipes(self):
        return [
            {
                "id": self.valves.DEEPSEEK_API_MODEL,
                "name": self.valves.DEEPSEEK_MODEL_DISPLAY_NAME,
            }
        ]

    def _search_searxng(self, query: str, results=None) -> str:
        if not self.valves.enable_searxng:
            return "SearXNG search is disabled"
        print("Searching with SearXNG")
        try:
            searxng_url = f"{self.valves.searxng_url}/search"
            params = {"q": query, "format": "json"}
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
            response = requests.get(searxng_url, params=params, headers=headers)
            response.raise_for_status()

            search_results = response.json()

            if not search_results.get("results"):
                return f"No results found for: {query}"

            formatted_results = f"SearXNG Search Results:\n\n"
            for i, result in enumerate(
                search_results["results"][
                    : results if results else self.valves.searxng_results
                ],
                1,
            ):
                title = result.get("title", "No title")
                snippet = result.get("content", "No snippet available")
                link = result.get("url", "No URL available")
                formatted_results += f"{i}. {title}\n   {snippet}\n   URL: {link}\n\n"

            return formatted_results

        except Exception as e:
            return f"An error occurred while searching SearXNG: {str(e)}"

    def _extract_current_query(self, body: dict) -> str:
        """Extract only the current turn's query without prior context."""
        messages = body.get("messages", [])
        if messages:
            last_message = messages[-1]
            if last_message["role"] == "user":  # Ensure it's a user message
                if isinstance(last_message.get("content"), list):
                    for item in last_message["content"]:
                        if item["type"] == "text":
                            return item["text"]
                else:
                    return last_message.get("content", "")
        return ""

    def _extract_user_input(self, body: dict) -> str:
        messages = body.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message.get("content"), list):
                for item in last_message["content"]:
                    if item["type"] == "text":
                        return item["text"]
            else:
                return last_message.get("content", "")
        return ""

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        """主处理管道（已移除缓冲）"""
        thinking_state = {"thinking": -1}  # 使用字典来存储thinking状态
        self.emitter = __event_emitter__
        user_input = self._extract_user_input(body)
        current_query = self._extract_current_query(body)
        # 验证配置
        if not self.valves.DEEPSEEK_API_KEY:
            yield json.dumps({"error": "未配置API密钥"}, ensure_ascii=False)
            return

        # 准备请求参数
        headers = {
            "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            # 模型ID提取
            model_id = body["model"].split(".", 1)[-1]
            payload = {**body, "model": model_id}

            # 处理消息以防止连续的相同角色
            messages = payload["messages"]

            # 添加 SearXNG 搜索逻辑

            search_input = None
            if current_query and self.valves.enable_searxng:
                search_input = self._search_searxng(current_query)
                if search_input and search_input != "SearXNG search is disabled":
                    if messages and messages[-1]["role"] == "user":
                        messages[-1][
                            "content"
                        ] += f"\n\nSearXNG Search Results:\n{search_input}"
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"{current_query}\n\nSearXNG Search Results:\n{search_input}",
                            }
                        )
            payload["messages"] = messages  # 更新 payload 中的 messages

            i = 0
            while i < len(messages) - 1:
                if messages[i]["role"] == messages[i + 1]["role"]:
                    # 插入具有替代角色的占位符消息
                    alternate_role = (
                        "assistant" if messages[i]["role"] == "user" else "user"
                    )
                    messages.insert(
                        i + 1,
                        {"role": alternate_role, "content": "[Unfinished thinking]"},
                    )
                i += 1

            async with httpx.AsyncClient(http2=True) as client:
                # 判断是否为流式请求
                if payload.get("stream", False):
                    # ========== 流式处理 ==========
                    async with client.stream(
                        "POST",
                        f"{self.valves.DEEPSEEK_API_BASE_URL}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=300,
                    ) as response:
                        if response.status_code != 200:
                            error = await response.aread()
                            yield self._format_error(response.status_code, error)
                            return

                        add_think_tag = False
                        async for line in response.aiter_lines():
                            if not line.startswith(self.data_prefix):
                                continue

                            if line.strip() == "data: [DONE]":
                                return

                            try:
                                data = json.loads(line[len(self.data_prefix) :])
                            except json.JSONDecodeError as e:
                                error_detail = f"解析失败 - 内容：{line}，原因：{e}"
                                yield self._format_error(
                                    "JSONDecodeError", error_detail
                                )
                                return

                            choice = data.get("choices", [{}])[0]
                            if choice.get("finish_reason"):
                                return

                            # 状态机处理
                            state_output = await self._update_thinking_state(
                                choice.get("delta", {}), thinking_state
                            )
                            if state_output:
                                yield state_output
                                if state_output == "<think>":
                                    yield "\n"

                            # 内容处理
                            content = self._process_content(choice["delta"])
                            if content:
                                if content.startswith("<think>"):
                                    add_think_tag = True
                                    content = content.replace("<think>", "", 1)
                                    yield "<think>"
                                    await asyncio.sleep(0.1)
                                    yield "\n"
                                elif not add_think_tag:
                                    add_think_tag = True
                                    yield "<think>"
                                    await asyncio.sleep(0.1)
                                    yield "\n"
                                elif content.startswith("</think>"):
                                    content = content.replace("</think>", "", 1)
                                    yield "</think>"
                                    await asyncio.sleep(0.1)
                                    yield "\n"
                                yield content
                else:
                    # ========== 非流式处理 ==========
                    response = await client.post(
                        f"{self.valves.DEEPSEEK_API_BASE_URL}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=300,
                    )

                    if response.status_code != 200:
                        error = response.content
                        yield self._format_error(response.status_code, error)
                        return

                    data = response.json()
                    # 直接提取最终回复内容（忽略思维链）
                    content = data["choices"][0]["message"].get("content", "")
                    yield content

        except Exception as e:
            yield self._format_exception(e)

    async def _update_thinking_state(self, delta: dict, thinking_state: dict) -> str:
        """更新思考状态机（简化版）"""
        state_output = ""

        # 状态转换：未开始 -> 思考中
        if thinking_state["thinking"] == -1 and delta.get("reasoning_content"):
            thinking_state["thinking"] = 0
            state_output = "<think>"

        # 状态转换：思考中 -> 已回答
        elif (
            thinking_state["thinking"] == 0
            and not delta.get("reasoning_content")
            and delta.get("content")
        ):
            thinking_state["thinking"] = 1
            state_output = "\n</think>\n\n"

        return state_output

    def _process_content(self, delta: dict) -> str:
        """直接返回处理后的内容"""
        return delta.get("reasoning_content", "") or delta.get("content", "")

    def _format_error(self, status_code: int, error: bytes) -> str:
        # 如果 error 已经是字符串，则无需 decode
        if isinstance(error, str):
            error_str = error
        else:
            error_str = error.decode(errors="ignore")

        try:
            err_msg = json.loads(error_str).get("message", error_str)[:200]
        except Exception as e:
            err_msg = error_str[:200]
        return json.dumps(
            {"error": f"HTTP {status_code}: {err_msg}"}, ensure_ascii=False
        )

    def _format_exception(self, e: Exception) -> str:
        """异常格式化保持不变"""
        err_type = type(e).__name__
        return json.dumps({"error": f"{err_type}: {str(e)}"}, ensure_ascii=False)
