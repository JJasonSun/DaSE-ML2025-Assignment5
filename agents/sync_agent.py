from typing import List, Dict, Optional
import tiktoken
import re
import requests
import json
import time

from model import ModelProvider


class SyncRetrievalAgent(ModelProvider):
    """
    同步版多文档检索 Agent，避免 asyncio 问题。
    使用 requests 替代 AsyncOpenAI，规避事件循环限制。
    
    提升准确率的优化点：
    1. 增强关键词提取（数字、日期、编码）
    2. 更好的文件打分算法
    3. 句子级内容抽取
    4. 多策略组合搜索
    """

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.model_name = "ecnu-plus"
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.tokenizer = tiktoken.encoding_for_model("gpt-4") # 使用 GPT-4 的编码器
        self.max_tokens_per_request = 10000  # 提高上限以覆盖更多上下文
        self.top_k_files = 5  # 从 3 提升至 5

    async def evaluate_model(self, prompt: Dict) -> str:
        """
        通过同步 HTTP 请求完成多文档检索。
        """
        try:
            context_data = prompt['context_data']
            question = prompt['question']

            # 提取相关内容
            selected_content = self._retrieve_content(context_data, question)

            # 构造消息
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Answer the question based on the provided context. If you find relevant information, provide a clear and accurate answer. If the information is not sufficient, say so clearly."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{selected_content}\n\nQuestion: {question}\n\nAnswer:"
                }
            ]

            # 发起同步 API 调用
            return self._make_sync_api_call(messages)
                
        except Exception as e:
            return f"Processing error: {str(e)[:50]}"

    def _make_sync_api_call(self, messages: List[Dict]) -> str:
        """
        使用同步请求避免 asyncio 相关问题。
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"API error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Request timed out."
        except Exception as e:
            return f"API call error: {str(e)[:50]}"

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
                if remaining > 200:  # 仅在剩余空间足够时追加
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
        months = re.findall(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', question.lower())
        keywords.extend(months)
        
        # 提取常规单词（编码提取之后）
        words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
        keywords.extend([w for w in words if w not in stop_words and len(w) > 2])
        
        # 去重并限制数量
        return list(dict.fromkeys(keywords))[:15]

    def _extract_relevant_content(self, content: str, keywords: List[str], question: str) -> str:
        """
        使用多种策略提取最相关的内容。
        """
        # 策略 1：找到包含关键词的句子
        sentences = re.split(r'[.!?]+', content)
        scored_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # 跳过过短的句子
                continue
                
            sentence_lower = sentence.lower()
            score = 0
            
            # 根据关键词出现次数得分
            for keyword in keywords:
                kw_lower = keyword.lower()
                if kw_lower in sentence_lower:
                    # 精确词边界额外加分
                    if re.search(r'\b' + re.escape(kw_lower) + r'\b', sentence_lower):
                        score += 3
                    else:
                        score += 1
            
            # 包含数字或日期给予加分
            if re.search(r'\b\d+\b', sentence):
                score += 1
            
            # 针对常见包含答案的模式再加分
            if re.search(r'\b(scheduled|date|day|week|days|between|from|to|on|in)\b', sentence_lower):
                score += 1
                
            if score > 0:
                scored_sentences.append((sentence, score))
        
        # 策略 2：若无高分句子，回退到段落级
        if not scored_sentences:
            return self._get_best_chunk(content, keywords)
        
        # 按得分排序并组合高分句子
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # 将高分句与上下文拼接
        result_sentences = []
        total_tokens = 0
        max_tokens = self.max_tokens_per_request // 6  # 为多个文件预留空间
        
        for sentence, score in scored_sentences[:10]:  # 取前 10 个句子
            sentence_tokens = len(self.encode_text_to_tokens(sentence))
            if total_tokens + sentence_tokens > max_tokens:
                break
            result_sentences.append(sentence)
            total_tokens += sentence_tokens
        
        # 若已找到较优句子，补充上下文
        if result_sentences:
            # 查找原文位置并加入周边内容
            result_with_context = []
            for target_sentence in result_sentences[:5]:  # 仅取前 5 个
                # 在原文中定位句子并添加前后文
                sentence_pos = content.find(target_sentence)
                if sentence_pos != -1:
                    # 获取前后一定范围的上下文
                    start = max(0, sentence_pos - 200)
                    end = min(len(content), sentence_pos + len(target_sentence) + 200)
                    context_chunk = content[start:end].strip()
                    result_with_context.append(context_chunk)
            
            return '\n\n'.join(result_with_context)
        
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
        max_tokens = self.max_tokens_per_request // 6
        
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
