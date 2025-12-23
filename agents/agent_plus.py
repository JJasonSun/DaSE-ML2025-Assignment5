import json
import os
import asyncio
from typing import List, Dict, Optional, Union
import tiktoken
import re
import requests
from openai import OpenAI
from dotenv import load_dotenv
from model import ModelProvider

class AdvancedRetrievalAgent(ModelProvider):
    """
    高级多文档检索 Agent。
    集成 ecnu-embedding-small 和 ecnu-rerank 模型提升检索精度。
    
    提升准确率的优化点：
    1. 增强关键词提取（数字、日期、编码）
    2. 初始关键词打分过滤文件
    3. 句子级片段抽取
    4. 使用 ecnu-rerank 对片段进行二次精排
    5. 32K 超大上下文窗口
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        load_dotenv()
        # 1. 主模型配置 (用于 LLM 生成)
        # 优先使用传入参数，否则按 API_KEY -> ECNU_API_KEY 顺序获取
        self.api_key = api_key or os.getenv('API_KEY') or os.getenv('ECNU_API_KEY')
        self.base_url = (base_url or os.getenv('BASE_URL') or os.getenv('ECNU_BASE_URL')).rstrip('/')
        self.model_name = os.getenv('MODEL_NAME', 'ecnu-max')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # 2. ECNU 专用模型配置 (用于 Embedding 和 Rerank)
        self.ecnu_api_key = os.getenv('ECNU_API_KEY') or self.api_key
        self.ecnu_base_url = os.getenv('ECNU_BASE_URL').rstrip('/')
        self.ecnu_client = OpenAI(api_key=self.ecnu_api_key, base_url=self.ecnu_base_url)
        
        self.embedding_model = "ecnu-embedding-small"
        self.rerank_model = "ecnu-rerank"
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-4") # 使用 GPT-4 的编码器
        self.max_tokens_per_request = 16000  # 有了 Rerank，16K 足以容纳高质量片段
        self.top_k_files = 15  # 初始筛选文件数
        
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
            unsupported_models = ['ecnu-max', 'ecnu-plus', 'ecnu-turbo']
            supports_thinking = not any(m in self.model_name.lower() for m in unsupported_models)
            
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
                **kwargs
            }

            # 如果模型支持 thinking 参数，根据环境变量显式开启或关闭
            if supports_thinking:
                if "extra_body" not in params:
                    params["extra_body"] = {}
                
                if enable_think:
                    params["extra_body"]["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": 1024 # 进一步压缩思维链长度以提升速度
                    }
                else:
                    # 默认显式禁用，防止某些模型（如 GLM-4.5）在某些情况下自动开启
                    params["extra_body"]["thinking"] = {
                        "type": "disabled"
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
            
            # 1. 检查是否有 native reasoning_content 并打印 (仅在开启思考模式时显示)
            reasoning = message.get('reasoning_content', '')
            enable_think = os.getenv('ENABLE_THINK', 'false').lower() == 'true'
            if isinstance(reasoning, str) and reasoning.strip() and enable_think:
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

    def _rerank_documents(self, query: str, documents: List[str], top_n: int = 10) -> List[str]:
        """使用 ecnu-rerank 模型对文档进行重排序。"""
        if not documents:
            return []
            
        # 强制使用 ECNU 专用的 URL 和 Key
        url = f"{self.ecnu_base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.ecnu_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            # 提取重排后的文档内容
            results = result.get('results', [])
            if not results:
                return documents[:top_n]
            
            reranked_docs = [item['document'] for item in results]
            return reranked_docs
        except Exception as e:
            print(f"Error during reranking: {e}")
            # 如果重排失败，返回原始文档的前 top_n 个
            return documents[:top_n]

    def _retrieve_content(self, context_data: Dict, question: str) -> str:
        """
        优化后的检索策略：全量分块 + 向量/关键词混合筛选 + 大规模 Rerank
        """
        files = context_data['files']
        keywords = self._get_keywords(question)
        query_emb = self._get_embeddings(question)
        
        all_chunks = []
        chunk_size = 1200
        overlap = 300
        
        # 1. 全量分块
        for file_data in files:
            filename = file_data['filename']
            content = file_data['modified_content']
            
            # 简单的滑动窗口分块
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk_text = content[start:end].strip()
                if len(chunk_text) > 50:
                    all_chunks.append({
                        "filename": filename,
                        "content": chunk_text,
                        "kw_score": 0.0,
                        "emb_score": 0.0
                    })
                start += (chunk_size - overlap)
        
        if not all_chunks:
            return "No relevant content found."

        # 2. 计算所有分块的得分 (并行/批量处理思想)
        # 注意：为了保证速度，我们对所有块进行关键词评分，对前 N 个进行向量评分
        for chunk in all_chunks:
            content_lower = chunk["content"].lower()
            # 关键词评分
            chunk["kw_score"] = sum(1 for kw in keywords if kw.lower() in content_lower)
        
        # 按关键词初步排序，取前 200 个进行向量精筛（如果 query_emb 存在）
        if query_emb:
            all_chunks.sort(key=lambda x: x["kw_score"], reverse=True)
            # 批量获取前 200 个块的向量
            candidate_chunks = all_chunks[:200]
            texts_to_embed = [c["content"] for c in candidate_chunks]
            
            # 分批处理，防止单次请求过大（虽然 200 应该没问题，但分批更稳）
            batch_size = 50
            all_doc_embs = []
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i+batch_size]
                batch_embs = self._get_embeddings(batch)
                all_doc_embs.extend(batch_embs)
            
            # 计算相似度
            for chunk, doc_emb in zip(candidate_chunks, all_doc_embs):
                if doc_emb:
                    chunk["emb_score"] = sum(a * b for a, b in zip(query_emb, doc_emb))
        
        # 3. 综合排序，选取候选池
        # 综合得分 = 归一化关键词(0.3) + 向量(0.7)
        for chunk in all_chunks:
            norm_kw = min(1.0, chunk["kw_score"] / (len(keywords) + 1))
            chunk["total_score"] = norm_kw * 0.3 + chunk["emb_score"] * 0.7
            
        all_chunks.sort(key=lambda x: x["total_score"], reverse=True)
        
        # 选取前 100 个最相关的片段进入 Rerank 阶段
        top_candidates = [f"[{c['filename']}] {c['content']}" for c in all_chunks[:100]]
        
        # 4. 大规模 Rerank (ecnu-rerank)
        # 既然没有上限，我们直接重排前 100 个片段
        reranked_snippets = self._rerank_documents(question, top_candidates, top_n=30)
        
        # 5. 组合最终上下文
        content_parts = []
        total_tokens = 0
        
        for snippet in reranked_snippets:
            snippet_tokens = len(self.encode_text_to_tokens(snippet))
            if total_tokens + snippet_tokens > self.max_tokens_per_request:
                break
            content_parts.append(snippet)
            total_tokens += snippet_tokens
            
        return "\n\n---\n\n".join(content_parts)

    def _extract_relevant_snippets(self, content: str, keywords: List[str]) -> List[str]:
        """提取包含关键词的候选片段。"""
        # 改进的句子切分
        abbr_2 = r'Mr|Ms|Dr|vs|St|Rd|Co'
        abbr_3 = r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Mrs|Ave|Inc|Ltd|Vol'
        sentence_endings = rf'(?<!\b(?:{abbr_2}))(?<!\b(?:{abbr_3}))\.(?!\d)|[.!?]+'
        
        try:
            sentences = re.split(sentence_endings, content)
        except Exception:
            sentences = re.split(r'[.!?]+', content)
            
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10: continue
            
            score = sum(10 if re.search(r'\b' + re.escape(kw.lower()) + r'\b', sentence.lower()) else 0 for kw in keywords)
            if score > 0:
                scored_sentences.append((i, sentence, score))
        
        if not scored_sentences:
            return [content[:1000]] # 兜底
            
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        
        snippets = []
        seen_ranges = []
        
        for idx, sentence, score in scored_sentences[:15]:
            # 找到句子在原文中的位置
            pos = content.find(sentence)
            if pos == -1: continue
            
            # 扩大上下文范围
            start = max(0, pos - 800)
            end = min(len(content), pos + len(sentence) + 800)
            
            # 检查是否重叠
            is_overlap = False
            for r_start, r_end in seen_ranges:
                if not (end < r_start or start > r_end):
                    is_overlap = True
                    break
            
            if not is_overlap:
                snippets.append(content[start:end].strip())
                seen_ranges.append((start, end))
                
        return snippets

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

    def _get_embeddings(self, input_data: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """使用 ecnu-embedding-small 获取文本向量，支持批量。"""
        try:
            # 使用 ECNU 专用客户端
            response = self.ecnu_client.embeddings.create(
                model=self.embedding_model,
                input=input_data
            )
            if isinstance(input_data, str):
                return response.data[0].embedding
            else:
                # 确保返回顺序一致
                embeddings = [None] * len(input_data)
                for item in response.data:
                    embeddings[item.index] = item.embedding
                return embeddings
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return [] if isinstance(input_data, str) else [[] for _ in input_data]

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
