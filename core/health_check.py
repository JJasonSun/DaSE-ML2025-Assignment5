def check_models(api_key: str, base_url: str, evaluator_type: str):
    import requests
    from openai import OpenAI
    import os

    from core.ecnu_constants import (
        DEFAULT_ECNU_BASE_URL,
        ECNU_EMBEDDING_MODEL_NAME,
        ECNU_MAIN_MODEL_NAME,
        ECNU_PLUS_MODEL_NAME,
        ECNU_RERANK_MODEL_NAME,
    )

    print("\n" + "-" * 80)
    print("[Health Check] Testing models availability before run...")
    
    ecnu_api_key = os.getenv('ECNU_API_KEY') or api_key
    ecnu_base_url = (os.getenv('ECNU_BASE_URL') or base_url or DEFAULT_ECNU_BASE_URL).rstrip('/')
    
    main_model_name = os.getenv('MODEL_NAME', ECNU_MAIN_MODEL_NAME)
    
    # 1. Main model
    main_client = OpenAI(api_key=api_key, base_url=base_url)
    main_client.chat.completions.create(
        model=main_model_name,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5
    )
    print(f"  [OK] Main Model ({main_model_name})")
        
    # 2. Evaluator model
    if evaluator_type == 'llm':
        eval_client = OpenAI(api_key=ecnu_api_key, base_url=ecnu_base_url)
        eval_client.chat.completions.create(
            model=ECNU_PLUS_MODEL_NAME,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5
        )
        print(f"  [OK] Evaluator Model ({ECNU_PLUS_MODEL_NAME})")

    # 3. ECNU Embedding
    ecnu_client = OpenAI(api_key=ecnu_api_key, base_url=ecnu_base_url)
    ecnu_client.embeddings.create(input="hello", model=ECNU_EMBEDDING_MODEL_NAME)
    print(f"  [OK] Embedding Model ({ECNU_EMBEDDING_MODEL_NAME})")

    # 4. ECNU Rerank
    url = f"{ecnu_base_url}/rerank"
    headers = {"Authorization": f"Bearer {ecnu_api_key}", "Content-Type": "application/json"}
    payload = {"model": ECNU_RERANK_MODEL_NAME, "query": "hi", "documents": ["hello"], "top_n": 1}
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    print(f"  [OK] Rerank Model ({ECNU_RERANK_MODEL_NAME})")
        
    print("-" * 80 + "\n")