def check_models(api_key: str, base_url: str, evaluator_type: str):
    import os

    import requests
    from openai import OpenAI, OpenAIError

    from core.ecnu_constants import (
        DEFAULT_ECNU_BASE_URL,
        ECNU_EMBEDDING_MODEL_NAME,
        ECNU_MAIN_MODEL_NAME,
        ECNU_PLUS_MODEL_NAME,
        ECNU_RERANK_MODEL_NAME,
    )

    def run_check(label: str, action):
        try:
            action()
        except (OpenAIError, requests.RequestException) as exc:
            print(f"  [FAIL] {label}")
            raise RuntimeError(
                f"Health check failed at {label}: {exc}. "
                "This usually means the upstream model service returned an error, "
                "the configured base URL/API key is invalid, or the model is temporarily unavailable."
            ) from exc
        print(f"  [OK] {label}")

    print("\n" + "-" * 80)
    print("[Health Check] Testing models availability before run...")

    ecnu_api_key = os.getenv("ECNU_API_KEY") or api_key
    ecnu_base_url = (os.getenv("ECNU_BASE_URL") or base_url or DEFAULT_ECNU_BASE_URL).rstrip("/")
    main_model_name = os.getenv("MODEL_NAME", ECNU_MAIN_MODEL_NAME)

    main_client = OpenAI(api_key=api_key, base_url=base_url)
    run_check(
        f"Main Model ({main_model_name})",
        lambda: main_client.chat.completions.create(
            model=main_model_name,
            messages=[{"role": "user", "content": "你好"}],
        ),
    )

    if evaluator_type == "llm":
        eval_client = OpenAI(api_key=ecnu_api_key, base_url=ecnu_base_url)
        run_check(
            f"Evaluator Model ({ECNU_PLUS_MODEL_NAME})",
            lambda: eval_client.chat.completions.create(
                model=ECNU_PLUS_MODEL_NAME,
                messages=[{"role": "user", "content": "你好"}],
            ),
        )

    ecnu_client = OpenAI(api_key=ecnu_api_key, base_url=ecnu_base_url)
    run_check(
        f"Embedding Model ({ECNU_EMBEDDING_MODEL_NAME})",
        lambda: ecnu_client.embeddings.create(input="hello", model=ECNU_EMBEDDING_MODEL_NAME),
    )

    url = f"{ecnu_base_url}/rerank"
    headers = {"Authorization": f"Bearer {ecnu_api_key}", "Content-Type": "application/json"}
    payload = {"model": ECNU_RERANK_MODEL_NAME, "query": "hi", "documents": ["hello"], "top_n": 1}
    run_check(
        f"Rerank Model ({ECNU_RERANK_MODEL_NAME})",
        lambda: requests.post(url, headers=headers, json=payload, timeout=10).raise_for_status(),
    )

    print("-" * 80 + "\n")
