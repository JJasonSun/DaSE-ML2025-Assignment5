#!/usr/bin/env python3
"""
测试LLMEvaluator是否正常工作
"""

import os
from dotenv import load_dotenv
from evaluators.llm_evaluator import LLMEvaluator

def test_evaluator():
    load_dotenv()
    
    api_key = os.getenv('API_KEY')
    base_url = os.getenv('BASE_URL')
    
    if not api_key or not base_url:
        print("❌ 缺少API配置")
        return
    
    # 创建评测器
    evaluator = LLMEvaluator(
        api_key=api_key,
        base_url=base_url,
        ground_truth="Thursday",
        question="What day of the week will the quantum encryption network go live?"
    )
    
    print("🔍 测试LLMEvaluator...")
    print(f"标准答案: Thursday")
    print(f"模型回答: Thursday")
    
    try:
        score = evaluator.evaluate_response("Thursday")
        print(f"✅ 评分: {score}/10")
        
        if score >= 8:
            print("✅ 评测器工作正常")
        else:
            print(f"⚠️ 评分偏低: {score}/10")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluator()
