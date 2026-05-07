def check_models(api_key: str, base_url: str, evaluator_type: str):
    import requests
    from openai import OpenAI
    import os

    print("\n" + "-" * 80)
    print("[Health Check] Testing models availability before run...")
    
    ecnu_api_key = os.getenv('ECNU_API_KEY') or api_key
    ecnu_base_url = (os.getenv('ECNU_BASE_URL') or base_url).rstrip('/')
    
    main_model_name = os.getenv('MODEL_NAME', 'Unknown')
    
    # 1. Main model
    try:
        main_client = OpenAI(api_key=api_key, base_url=base_url)
        main_client.chat.completions.create(
            model=main_model_name,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5
        )
        print(f"  [OK] Main Model ({main_model_name})")
    except Exception as e:
        print(f"  [FAIL] Main Model ({main_model_name}): {e}")
        
    # 2. Evaluator model
    if evaluator_type == 'llm':
        eval_api_key = os.getenv('EVAL_API_KEY') or api_key
        eval_base_url = os.getenv('EVAL_BASE_URL') or base_url
        eval_model_name = os.getenv('EVAL_MODEL_NAME', main_model_name)
        try:
            eval_client = OpenAI(api_key=eval_api_key, base_url=eval_base_url)
            eval_client.chat.completions.create(
                model=eval_model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5
            )
            print(f"  [OK] Evaluator Model ({eval_model_name})")
        except Exception as e:
            print(f"  [FAIL] Evaluator Model ({eval_model_name}): {e}")

    # 3. ECNU Embedding
    try:
        ecnu_client = OpenAI(api_key=ecnu_api_key, base_url=ecnu_base_url)
        ecnu_client.embeddings.create(input="hello", model="ecnu-embedding-small")
        print("  [OK] Embedding Model (ecnu-embedding-small)")
    except Exception as e:
        print(f"  [FAIL] Embedding Model (ecnu-embedding-small): {e}")

    # 4. ECNU Rerank
    try:
        url = f"{ecnu_base_url}/rerank"
        headers = {"Authorization": f"Bearer {ecnu_api_key}", "Content-Type": "application/json"}
        payload = {"model": "ecnu-rerank", "query": "hi", "documents": ["hello"], "top_n": 1}
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        print("  [OK] Rerank Model (ecnu-rerank)")
    except Exception as e:
        print(f"  [FAIL] Rerank Model (ecnu-rerank): {e}")
        
    print("-" * 80 + "\n")