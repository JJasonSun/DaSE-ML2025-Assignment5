import os
import sys
from dotenv import load_dotenv

# 将项目根目录添加到路径，以便导入 evaluators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.llm_evaluator import LLMEvaluator

def test_evaluator():
    load_dotenv()
    
    # 获取评测配置
    api_key = os.getenv('EVAL_API_KEY') or os.getenv('API_KEY')
    base_url = os.getenv('EVAL_BASE_URL') or os.getenv('BASE_URL')
    model_name = os.getenv('EVAL_MODEL_NAME') or os.getenv('MODEL_NAME')
    
    if not api_key or not base_url:
        print("Error: API_KEY or BASE_URL not found in .env")
        return

    print(f"Testing Evaluator with model: {model_name}")
    print(f"Base URL: {base_url}")
    print("-" * 50)

    test_scenarios = [
        {
            "name": "Perfect Match",
            "question": "What day of the week is December 25, 2031?",
            "ground_truth": "Thursday",
            "candidate": "Thursday"
        },
        {
            "name": "Wrong Fact (Should be 0)",
            "question": "What day of the week is December 25, 2031?",
            "ground_truth": "Thursday",
            "candidate": "Friday"
        },
        {
            "name": "Partial Match / Extra Text",
            "question": "What day of the week is December 25, 2031?",
            "ground_truth": "Thursday",
            "candidate": "Based on the deployment schedule, it is Thursday."
        },
        {
            "name": "Cannot Generate (Should be 0)",
            "question": "What day of the week is December 25, 2031?",
            "ground_truth": "Thursday",
            "candidate": "I cannot find this information in the context."
        },
        {
            "name": "Numerical Match",
            "question": "How many system integration tests were performed?",
            "ground_truth": "42",
            "candidate": "42"
        },
        {
            "name": "Wrong Number (Should be 0)",
            "question": "How many system integration tests were performed?",
            "ground_truth": "42",
            "candidate": "43"
        }
    ]

    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Question: {scenario['question']}")
        print(f"Ground Truth: {scenario['ground_truth']}")
        print(f"Candidate: {scenario['candidate']}")
        
        evaluator = LLMEvaluator(
            api_key=api_key,
            base_url=base_url,
            ground_truth=scenario['ground_truth'],
            question=scenario['question']
        )
        
        score = evaluator.evaluate_response(scenario['candidate'])
        print(f"Resulting Score: {score}/10")
        
        # 简单的逻辑验证
        if scenario['name'] == "Wrong Fact (Should be 0)" and score > 0:
            print("❌ WARNING: Wrong fact should have scored 0!")
        elif scenario['name'] == "Perfect Match" and score < 10:
            print("⚠️ NOTE: Perfect match scored less than 10.")
        elif scenario['name'] == "Cannot Generate (Should be 0)" and score > 0:
            print("❌ WARNING: 'Cannot generate' should have scored 0!")
        else:
            print("✅ Logic seems consistent.")

if __name__ == "__main__":
    test_evaluator()
