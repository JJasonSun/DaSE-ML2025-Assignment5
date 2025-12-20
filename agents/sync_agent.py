from typing import List, Dict, Optional
import tiktoken
import re
import requests
import json
import time

from model import ModelProvider


class SyncRetrievalAgent(ModelProvider):
    """
    Synchronous multi-document retrieval agent to avoid asyncio issues.
    Uses requests instead of AsyncOpenAI to eliminate event loop problems.
    
    Optimizations for better accuracy:
    1. Enhanced keyword extraction (numbers, dates, codes)
    2. Better file scoring algorithm
    3. Sentence-level content extraction
    4. Multiple search strategies
    """

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.model_name = "ecnu-plus"
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.tokenizer = tiktoken.encoding_for_model("gpt-4") # 使用 GPT-4 的编码器
        self.max_tokens_per_request = 10000  # Increased for better coverage
        self.top_k_files = 5  # Increased from 3 to 5

    async def evaluate_model(self, prompt: Dict) -> str:
        """
        Handle multi-document retrieval using synchronous HTTP requests.
        """
        try:
            context_data = prompt['context_data']
            question = prompt['question']

            # Extract relevant content
            selected_content = self._retrieve_content(context_data, question)

            # Prepare messages
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

            # Make synchronous API call
            return self._make_sync_api_call(messages)
                
        except Exception as e:
            return f"Processing error: {str(e)[:50]}"

    def _make_sync_api_call(self, messages: List[Dict]) -> str:
        """
        Make a synchronous API call to avoid asyncio issues.
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
        Enhanced content retrieval with better scoring and extraction.
        """
        files = context_data['files']
        
        # Extract keywords
        keywords = self._get_keywords(question)
        
        # Find relevant files with enhanced scoring
        relevant_files = []
        for file_data in files:
            filename = file_data['filename']
            content = file_data['modified_content']
            content_lower = content.lower()
            
            # Multiple scoring factors
            keyword_matches = 0
            exact_matches = 0
            
            for keyword in keywords:
                kw_lower = keyword.lower()
                # Count all occurrences
                keyword_matches += content_lower.count(kw_lower)
                # Bonus for exact word boundaries
                if re.search(r'\b' + re.escape(kw_lower) + r'\b', content_lower):
                    exact_matches += 1
            
            if keyword_matches > 0 or exact_matches > 0:
                # Enhanced scoring: keyword density + exact match bonus
                content_length = len(content.split())
                density_score = keyword_matches / max(content_length, 1) * 1000
                exact_bonus = exact_matches * 100
                final_score = density_score + exact_bonus
                
                relevant_files.append((filename, final_score, file_data))
        
        # Sort by relevance and take top files
        relevant_files.sort(key=lambda x: x[1], reverse=True)
        
        # Extract content from top files
        content_parts = []
        total_tokens = 0
        
        for filename, score, file_data in relevant_files[:self.top_k_files]:
            if total_tokens >= self.max_tokens_per_request:
                break
                
            content = file_data['modified_content']
            
            # Get best content from this file
            extracted_content = self._extract_relevant_content(content, keywords, question)
            
            # Check token limit
            content_tokens = len(self.encode_text_to_tokens(extracted_content))
            if total_tokens + content_tokens > self.max_tokens_per_request:
                remaining = self.max_tokens_per_request - total_tokens
                if remaining > 200:  # Only add if meaningful space
                    extracted_content = self._truncate_text(extracted_content, remaining)
                    content_parts.append(f"=== {filename} ===\n{extracted_content}")
                break
            else:
                content_parts.append(f"=== {filename} ===\n{extracted_content}")
                total_tokens += content_tokens
        
        return "\n\n".join(content_parts) if content_parts else "No relevant content found."

    def _get_keywords(self, question: str) -> List[str]:
        """Enhanced keyword extraction including numbers, dates, and codes."""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how',
            'who', 'which', 'that', 'this', 'these', 'those'
        }
        
        keywords = []
        
        # Extract alphanumeric codes FIRST (before lowercasing)
        codes = re.findall(r'\b[A-Z0-9]+-[A-Z0-9-]+[A-Z0-9]\b', question)
        keywords.extend([code.lower() for code in codes])
        
        # Extract project codes like P-8812-Cerulean
        project_codes = re.findall(r'\b[A-Z]-\d+-[A-Za-z]+\b', question)
        keywords.extend([code.lower() for code in project_codes])
        
        # Extract simple codes like AP-C7X9
        simple_codes = re.findall(r'\b[A-Z]{2,}-[A-Z0-9]+\b', question)
        keywords.extend([code.lower() for code in simple_codes])
        
        # Extract numbers (important for dates, codes, etc.)
        numbers = re.findall(r'\b\d+\b', question)
        keywords.extend(numbers)
        
        # Extract years (2020-2099)
        years = re.findall(r'\b20[2-9]\d\b', question)
        keywords.extend(years)
        
        # Extract month names
        months = re.findall(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', question.lower())
        keywords.extend(months)
        
        # Extract regular words (after extracting codes)
        words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
        keywords.extend([w for w in words if w not in stop_words and len(w) > 2])
        
        # Remove duplicates and limit
        return list(dict.fromkeys(keywords))[:15]

    def _extract_relevant_content(self, content: str, keywords: List[str], question: str) -> str:
        """
        Extract the most relevant content using multiple strategies.
        """
        # Strategy 1: Find sentences containing keywords
        sentences = re.split(r'[.!?]+', content)
        scored_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            score = 0
            
            # Score based on keyword presence
            for keyword in keywords:
                kw_lower = keyword.lower()
                if kw_lower in sentence_lower:
                    # Bonus for exact word boundaries
                    if re.search(r'\b' + re.escape(kw_lower) + r'\b', sentence_lower):
                        score += 3
                    else:
                        score += 1
            
            # Bonus for numbers and dates (often contain answers)
            if re.search(r'\b\d+\b', sentence):
                score += 1
            
            # Bonus for specific patterns that often contain answers
            if re.search(r'\b(scheduled|date|day|week|days|between|from|to|on|in)\b', sentence_lower):
                score += 1
                
            if score > 0:
                scored_sentences.append((sentence, score))
        
        # Strategy 2: If no good sentences, fall back to paragraphs
        if not scored_sentences:
            return self._get_best_chunk(content, keywords)
        
        # Sort sentences by score and combine top ones
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Combine top sentences with context
        result_sentences = []
        total_tokens = 0
        max_tokens = self.max_tokens_per_request // 6  # Reserve space for multiple files
        
        for sentence, score in scored_sentences[:10]:  # Top 10 sentences
            sentence_tokens = len(self.encode_text_to_tokens(sentence))
            if total_tokens + sentence_tokens > max_tokens:
                break
            result_sentences.append(sentence)
            total_tokens += sentence_tokens
        
        # If we have good sentences, add some context around them
        if result_sentences:
            # Find the original positions and add context
            result_with_context = []
            for target_sentence in result_sentences[:5]:  # Limit to top 5
                # Find sentence in original content and add surrounding context
                sentence_pos = content.find(target_sentence)
                if sentence_pos != -1:
                    # Get some context before and after
                    start = max(0, sentence_pos - 200)
                    end = min(len(content), sentence_pos + len(target_sentence) + 200)
                    context_chunk = content[start:end].strip()
                    result_with_context.append(context_chunk)
            
            return '\n\n'.join(result_with_context)
        
        return '\n'.join(result_sentences)

    def _get_best_chunk(self, content: str, keywords: List[str]) -> str:
        """Fallback method: Get the most relevant chunk from content."""
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return content[:1500]  # Fallback
        
        # Score paragraphs by keyword presence
        scored_paras = []
        for para in paragraphs:
            para_lower = para.lower()
            score = sum(para_lower.count(kw.lower()) for kw in keywords)
            if score > 0:
                scored_paras.append((para, score))
        
        if not scored_paras:
            # No keywords found, return first few paragraphs
            return '\n\n'.join(paragraphs[:2])
        
        # Sort by score and combine top paragraphs
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
        """Truncate text to token limit."""
        tokens = self.encode_text_to_tokens(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode_tokens(tokens[:max_tokens])

    def generate_prompt(self, **kwargs) -> Dict:
        """Generate prompt structure."""
        return {
            'context_data': kwargs.get('context_data'),
            'question': kwargs.get('question')
        }

    def encode_text_to_tokens(self, text: str) -> List[int]:
        """Encode text to tokens."""
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        """Decode tokens to text."""
        if context_length:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)