import json
import os
import asyncio
from typing import List, Dict, Optional
import tiktoken
import re
from openai import OpenAI
from dotenv import load_dotenv
from model import ModelProvider

class SyncRetrievalAgent(ModelProvider):
    """
    同步版多文档检索 Agent，避免 asyncio 问题。
    使用基类的通用API调用方法，支持控制推理过程。
    
    提升准确率的优化点：
    1. 增强关键词提取（数字、日期、编码）
    2. 更好的文件打分算法
    3. 句子级内容抽取
    4. 多策略组合搜索
    5. 禁用推理模式获得直接答案
    """

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        load_dotenv()
        self.model_name = os.getenv('MODEL_NAME', 'glm-4.5')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-4") # 使用 GPT-4 的编码器
        self.max_tokens_per_request = 32000  # 进一步增加上下文长度以提高召回率
        self.top_k_files = 20  # 增加检索文件数量以覆盖更多可能性
        
        # 加载增强提示词
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """从 JSON 文件加载提示词，如果失败则使用默认值。"""
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'agent_prompts.json')
        try:
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        
        # 默认提示词 (兜底)
        return {
            "system_prompt": (
                "You are a very strong reasoner and planner. Use these critical instructions to structure your plans, thoughts, and responses.\n\n"
                "Before taking any action, you must proactively, methodically, and independently plan and reason about:\n"
                "1. **Logical dependencies and constraints**: Analyze the question against the provided snippets.\n"
                "2. **Abductive reasoning and hypothesis exploration**: Perform semantic mapping if direct match is missing.\n"
                "3. **Risk assessment**: Avoid hallucinations. The 'needle' is guaranteed to exist.\n"
                "4. **Precision and Grounding**: Ensure reasoning is precise and relevant to snippets.\n"
                "5. **Persistence and patience**: Do not give up. Re-scan context if needed.\n\n"
                "### OUTPUT FORMAT:\n"
                "Return ONLY a valid JSON object: {\"answer\": \"...\"}."
            ),
            "user_prompt_template": "Context:\n{context}\n\nQuestion: {question}\n\nReturn your answer in JSON format: {{\"answer\": \"...\"}}"
        }

    async def _create_chat_completion(self,
                                      messages: List[Dict],
                                      temperature: float = 0,
                                      max_tokens: int = 500,
                                      timeout: int = 20,
                                      **kwargs) -> str:
        """在异步上下文中用 OpenAI 客户端调用 ChatCompletion 并返回文本。"""
        def _sync_call():
            # 从环境变量获取是否启用思考模式，默认为 false
            enable_think = os.getenv('ENABLE_THINK', 'false').lower() == 'true'
            
            # 只有 ecnu-max 和 ecnu-plus 不支持 thinking 参数
            unsupported_models = ['ecnu-max', 'ecnu-plus']
            supports_thinking = not any(m in self.model_name.lower() for m in unsupported_models)
            
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
                **kwargs
            }

            # 如果启用了思考模式且模型支持，添加 extra_body
            if enable_think and supports_thinking:
                params["extra_body"] = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024 # 进一步压缩思维链长度以提升速度
                    }
                }

            return self.client.chat.completions.create(**params)

        try:
            try:
                completion = await asyncio.to_thread(_sync_call)
            except AttributeError:
                loop = asyncio.get_running_loop()
                completion = await loop.run_in_executor(None, _sync_call)

            raw = completion.to_dict()
            print(f"Debug: Raw response: {raw}") # 调试输出完整响应
            return self._extract_content_from_response(raw)
        except Exception as exc:
            return f"API error: {exc}" if exc else "API error"

    def _extract_content_from_response(self, result: dict) -> str:
        """
        从API响应中提取内容，兼容多种格式。
        """
        try:
            choice = result.get('choices', [{}])[0]
            message = choice.get('message', {})
            
            # 1. 检查是否有 native reasoning_content 并打印
            reasoning = message.get('reasoning_content', '')
            if isinstance(reasoning, str) and reasoning.strip():
                print(f"Debug: Native Reasoning Content: {reasoning.strip()}")
            
            # 2. 优先获取常规 content (对于 JSON 模式，答案在这里)
            content = message.get('content', '')
            if isinstance(content, str) and content.strip():
                # 额外处理：有些模型会将思考内容放在 content 中并用特定标签包裹
                clean_content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL).strip()
                if clean_content:
                    return clean_content
                return content.strip()
            
            # 3. 如果 content 为空但有 reasoning，说明模型可能还没输出结果就中断了
            if isinstance(reasoning, str) and reasoning.strip():
                # 不直接返回 reasoning，因为那不是答案，返回空字符串以触发备用策略
                print(f"Warning: Model provided reasoning but no content. Finish reason: {choice.get('finish_reason')}")
                return ""
            
            # 4. 检查是否有工具调用
            tool_calls = message.get('tool_calls')
            if tool_calls:
                return f"Tool calls detected: {tool_calls}"
            
            # 4. 返回finish_reason信息
            finish_reason = choice.get('finish_reason', 'unknown')
            return f"Empty response (finish_reason: {finish_reason})"
            
        except Exception as e:
            return f"Response parsing error: {str(e)[:50]}"

    async def evaluate_model(self, prompt: Dict) -> str:
        """
        通过同步 HTTP 请求完成多文档检索。
        """
        try:
            context_data = prompt.get('context_data', {})
            question = prompt.get('question', '')
            
            if not context_data or not question:
                return "Missing required input data"

            # 提取相关内容
            selected_content = self._retrieve_content(context_data, question)
            
            if not selected_content or selected_content == "No relevant content found.":
                return "No relevant content found"
            
            # 调试：打印检索到的内容长度和前 200 字符
            print(f"Debug: Selected content length: {len(selected_content)} chars")
            if len(selected_content) > 0:
                # 检查 needle 是否在 selected_content 中 (仅用于调试)
                # 注意：我们不能读取原始文件，但我们可以检查 selected_content 是否包含问题中的关键编码
                keywords = self._get_keywords(question)
                found_kws = [kw for kw in keywords if kw.lower() in selected_content.lower()]
                print(f"Debug: Found {len(found_kws)}/{len(keywords)} keywords in selected content")

            # 使用加载的增强提示词
            system_prompt = self.prompts.get("system_prompt")
            user_template = self.prompts.get("user_prompt_template")
            user_content = user_template.format(context=selected_content, question=question)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            # 使用基类的API调用方法，优化参数
            response_raw = await self._create_chat_completion(
                messages=messages,
                temperature=0,
                max_tokens=8000, # 进一步增加 max_tokens 以容纳超长思考
                timeout=90,      # 增加超时时间
                response_format={"type": "json_object"}
            )
            
            # 解析 JSON 响应
            response = ""
            try:
                # 1. 预处理：剥离可能存在的 Markdown 代码块
                clean_raw = response_raw.strip()
                if "```" in clean_raw:
                    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_raw, re.DOTALL)
                    if json_match:
                        clean_raw = json_match.group(1)
                
                # 2. 尝试解析 JSON
                data = json.loads(clean_raw)
                response = str(data.get("answer", ""))
            except Exception:
                # 3. 兜底：如果 JSON 解析失败，尝试用正则提取
                # 尝试提取 answer 字段的值
                json_match = re.search(r'\"answer\"\s*:\s*\"(.*?)\"', response_raw, re.DOTALL)
                if json_match:
                    response = json_match.group(1)
                else:
                    response = response_raw

            # 如果响应为空或包含错误信息，尝试备用策略
            if not response or response.strip() == "" or "error" in response.lower() or "empty" in response.lower() or "cannot generate" in response.lower():
                # 尝试更严谨的备用 prompt
                backup_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Expert Retrieval Mode: Locate the 'needle' in the provided snippets. "
                            "The information IS present. Scan methodically. "
                            "Return ONLY a JSON object: {\"answer\": \"...\"}"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Snippets:\n{selected_content}\n\nTarget Question: {question}"
                    }
                ]
                backup_raw = await self._create_chat_completion(
                    messages=backup_messages,
                    temperature=0,
                    max_tokens=100,
                    timeout=15,
                    response_format={"type": "json_object"}
                )
                try:
                    # 同样对备用响应进行 Markdown 剥离
                    clean_backup = backup_raw.strip()
                    if "```" in clean_backup:
                        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_backup, re.DOTALL)
                        if json_match:
                            clean_backup = json_match.group(1)
                    data = json.loads(clean_backup)
                    response = str(data.get("answer", ""))
                except Exception:
                    response = backup_raw
            
            # 最终检查和清理响应
            if response and response.strip():
                cleaned_response = response.strip()
                
                # 移除可能的前缀（循环移除，直到没有匹配为止）
                prefixes_to_remove = [
                    'Answer:', 'The answer is', 'Based on', 'According to', 
                    'Result:', 'Output:', 'Final Answer:', '答案：'
                ]
                
                changed = True
                while changed:
                    changed = False
                    for prefix in prefixes_to_remove:
                        if cleaned_response.lower().startswith(prefix.lower()):
                            cleaned_response = cleaned_response[len(prefix):].strip()
                            if cleaned_response.startswith(':'):
                                cleaned_response = cleaned_response[1:].strip()
                            changed = True
                
                # 移除末尾的引号（有时 JSON 提取会残留）
                cleaned_response = cleaned_response.strip('"\'')
                
                # 移除句号等标点符号（如果答案很短，通常是单个词或日期）
                if len(cleaned_response.split()) <= 5:
                    cleaned_response = cleaned_response.rstrip('.,!?;')
                
                return cleaned_response
            else:
                return "I cannot generate a valid answer."
                
        except Exception as e:
            return f"Error: {str(e)[:50]}"

    def _retrieve_content(self, context_data: Dict, question: str) -> str:
        """
        使用改进的打分与抽取策略获取更相关的内容。
        """
        files = context_data['files']
        
        # 提取关键词
        keywords = self._get_keywords(question)
        
        # 通过强化评分寻找相关文件
        relevant_files = []
        for file_data in files:
            filename = file_data['filename']
            content = file_data['modified_content']
            content_lower = content.lower()
            
            # 多重评分因素
            keyword_matches = 0
            exact_matches = 0
            
            for keyword in keywords:
                kw_lower = keyword.lower()
                # 统计出现次数
                keyword_matches += content_lower.count(kw_lower)
                # 精确词边界给予额外加成
                if re.search(r'\b' + re.escape(kw_lower) + r'\b', content_lower):
                    exact_matches += 1
            
            if keyword_matches > 0 or exact_matches > 0:
                # 综合得分：关键词密度 + 精确匹配加分
                content_length = len(content.split())
                density_score = keyword_matches / max(content_length, 1) * 1000
                exact_bonus = exact_matches * 100
                final_score = density_score + exact_bonus
                
                relevant_files.append((filename, final_score, file_data))
        
        # 按相关性排序并选取前若干文件
        relevant_files.sort(key=lambda x: x[1], reverse=True)
        
        # 从高相关文件中提取内容
        content_parts = []
        total_tokens = 0
        
        for filename, score, file_data in relevant_files[:self.top_k_files]:
            if total_tokens >= self.max_tokens_per_request:
                break
                
            content = file_data['modified_content']
            
            # 获取该文件中最相关的片段
            extracted_content = self._extract_relevant_content(content, keywords, question)
            
            # 检查 token 上限
            content_tokens = len(self.encode_text_to_tokens(extracted_content))
            if total_tokens + content_tokens > self.max_tokens_per_request:
                remaining = self.max_tokens_per_request - total_tokens
                if remaining > 500:  # 增加阈值
                    extracted_content = self._truncate_text(extracted_content, remaining)
                    content_parts.append(f"=== {filename} ===\n{extracted_content}")
                break
            else:
                content_parts.append(f"=== {filename} ===\n{extracted_content}")
                total_tokens += content_tokens
        
        return "\n\n".join(content_parts) if content_parts else "No relevant content found."

    def _get_keywords(self, question: str) -> List[str]:
        """增强关键词提取，包含数字、日期和编码。"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how',
            'who', 'which', 'that', 'this', 'these', 'those'
        }
        
        keywords = []
        
        # 先提取字母数字编码（在转小写前）
        codes = re.findall(r'\b[A-Z0-9]+-[A-Z0-9-]+[A-Z0-9]\b', question)
        keywords.extend([code.lower() for code in codes])
        
        # 提取类似 P-8812-Cerulean 的项目编码
        project_codes = re.findall(r'\b[A-Z]-\d+-[A-Za-z]+\b', question)
        keywords.extend([code.lower() for code in project_codes])
        
        # 提取类似 AP-C7X9 的简短编码
        simple_codes = re.findall(r'\b[A-Z]{2,}-[A-Z0-9]+\b', question)
        keywords.extend([code.lower() for code in simple_codes])
        
        # 提取数字（日期、编码等常用）
        numbers = re.findall(r'\b\d+\b', question)
        keywords.extend(numbers)
        
        # 提取年份（2020-2099）
        years = re.findall(r'\b20[2-9]\d\b', question)
        keywords.extend(years)
        
        # 提取月份名称
        months = re.findall(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', question.lower())
        keywords.extend(months)

        # 提取星期名称
        days = re.findall(r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b', question.lower())
        keywords.extend(days)
        
        # 提取常规单词（编码提取之后）
        words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
        keywords.extend([w for w in words if w not in stop_words and len(w) > 2])
        
        # 去重并限制数量
        return list(dict.fromkeys(keywords))[:20]

    def _extract_relevant_content(self, content: str, keywords: List[str], question: str) -> str:
        """
        使用多种策略提取最相关的内容。
        """
        # 如果内容本身不长，直接返回全文以保证完整性
        if len(self.encode_text_to_tokens(content)) < 1500:
            return content

        # 策略 1：找到包含关键词的句子
        # 改进的句子切分：避免在日期（2024.01.01）或常见缩写处切断
        abbr_2 = r'Mr|Ms|Dr|vs|St|Rd|Co'
        abbr_3 = r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Mrs|Ave|Inc|Ltd|Vol'
        sentence_endings = rf'(?<!\b(?:{abbr_2}))(?<!\b(?:{abbr_3}))\.(?!\d)|[.!?]+'
        
        try:
            sentences = re.split(sentence_endings, content)
        except Exception:
            sentences = re.split(r'[.!?]+', content)
            
        scored_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            sentence_lower = sentence.lower()
            score = 0
            
            for keyword in keywords:
                kw_lower = keyword.lower()
                if kw_lower in sentence_lower:
                    if re.search(r'\b' + re.escape(kw_lower) + r'\b', sentence_lower):
                        score += 10 # 显著增加权重
                    else:
                        score += 3
            
            if re.search(r'\b\d+\b', sentence):
                score += 2
            
            if re.search(r'\b(scheduled|date|day|week|days|between|from|to|on|in|at|reference|today|tomorrow|yesterday)\b', sentence_lower):
                score += 5
                
            if score > 0:
                scored_sentences.append((sentence, score))
        
        if not scored_sentences:
            return self._get_best_chunk(content, keywords)
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        result_sentences = []
        total_tokens = 0
        max_tokens_per_file = self.max_tokens_per_request // 3 # 增加单文件预算
        
        for sentence, score in scored_sentences[:30]: # 增加候选
            sentence_tokens = len(self.encode_text_to_tokens(sentence))
            if total_tokens + sentence_tokens > max_tokens_per_file:
                break
            result_sentences.append(sentence)
            total_tokens += sentence_tokens
        
        if result_sentences:
            result_with_context = []
            sorted_sentences = []
            for s in result_sentences:
                pos = content.find(s)
                if pos != -1:
                    sorted_sentences.append((s, pos))
            sorted_sentences.sort(key=lambda x: x[1])

            for target_sentence, sentence_pos in sorted_sentences[:15]: # 增加片段数量
                # 扩大上下文范围到 800 字符
                start = max(0, sentence_pos - 800)
                end = min(len(content), sentence_pos + len(target_sentence) + 800)
                context_chunk = content[start:end].strip()
                if context_chunk not in result_with_context:
                    result_with_context.append(context_chunk)
            
            return '\n\n... [SNIPPET] ...\n\n'.join(result_with_context)
        
        return self._get_best_chunk(content, keywords)
        
        return self._get_best_chunk(content, keywords)
        
        return '\n'.join(result_sentences)

    def _get_best_chunk(self, content: str, keywords: List[str]) -> str:
        """回退策略：从内容中获取最相关的片段。"""
        # 按段落拆分
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return content[:1500]  # 兜底
        
        # 根据关键词出现次数为段落打分
        scored_paras = []
        for para in paragraphs:
            para_lower = para.lower()
            score = sum(para_lower.count(kw.lower()) for kw in keywords)
            if score > 0:
                scored_paras.append((para, score))
        
        if not scored_paras:
            # 未找到关键词则返回前若干段
            return '\n\n'.join(paragraphs[:2])
        
        # 按得分排序并组合高分段落
        scored_paras.sort(key=lambda x: x[1], reverse=True)
        
        result_paras = []
        total_tokens = 0
        max_tokens = self.max_tokens_per_request // 4
        
        for para, score in scored_paras:
            para_tokens = len(self.encode_text_to_tokens(para))
            if total_tokens + para_tokens > max_tokens:
                break
            result_paras.append(para)
            total_tokens += para_tokens
        
        return '\n\n'.join(result_paras) if result_paras else scored_paras[0][0]

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """将文本截断到 token 上限。"""
        tokens = self.encode_text_to_tokens(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode_tokens(tokens[:max_tokens])

    def generate_prompt(self, **kwargs) -> Dict:
        """生成 prompt 结构。"""
        return {
            'context_data': kwargs.get('context_data'),
            'question': kwargs.get('question')
        }

    def encode_text_to_tokens(self, text: str) -> List[int]:
        """将文本编码为 tokens。"""
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        """将 tokens 解码回文本。"""
        if context_length:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)
