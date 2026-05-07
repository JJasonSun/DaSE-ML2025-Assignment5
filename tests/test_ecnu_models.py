import os
import requests
import numpy as np
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ecnu_constants import (
    DEFAULT_ECNU_BASE_URL,
    ECNU_EMBEDDING_MODEL_NAME,
    ECNU_RERANK_MODEL_NAME,
)

def test_ecnu_models_niah():
    load_dotenv()
    
    api_key = os.getenv('ECNU_API_KEY')
    base_url = (os.getenv('ECNU_BASE_URL') or DEFAULT_ECNU_BASE_URL).rstrip('/')
    
    if not api_key:
        print("❌ Error: ECNU_API_KEY not found in .env")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    print(f"Testing ECNU Models (NIAH Simulation) at: {base_url}")
    print("-" * 60)

    # 1. Prepare Haystack
    essay_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PaulGrahamEssays")
    essays = ["addiction.txt", "apple.txt", "bias.txt", "boss.txt", "copy.txt"]
    haystack = []
    
    for essay in essays:
        path = os.path.join(essay_dir, essay)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                haystack.append({"title": essay, "content": f.read()[:2000]}) # Take first 2000 chars

    # 2. Insert Needle
    needle = "The secret ingredient for a perfect startup is a pinch of saffron and a dash of luck."
    needle_index = 2 # Insert into bias.txt
    haystack[needle_index]["content"] += "\n\n" + needle
    
    query = "What is the secret ingredient for a perfect startup?"
    
    print(f"Haystack size: {len(haystack)} documents")
    print(f"Needle inserted into: {haystack[needle_index]['title']}")
    print(f"Query: {query}")
    print("-" * 60)

    # 3. Step 1: Embedding Retrieval (Simulated)
    print(f"\n[Step 1] Embedding Retrieval ({ECNU_EMBEDDING_MODEL_NAME})...")
    try:
        # Get query embedding
        query_emb = client.embeddings.create(
            model=ECNU_EMBEDDING_MODEL_NAME,
            input=query
        ).data[0].embedding
        
        # Get doc embeddings and calculate cosine similarity
        scores = []
        for doc in haystack:
            doc_emb = client.embeddings.create(
                model=ECNU_EMBEDDING_MODEL_NAME,
                input=doc["content"]
            ).data[0].embedding
            
            # Cosine similarity
            similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            scores.append(similarity)
        
        # Sort by embedding score
        emb_results = sorted(zip(haystack, scores), key=lambda x: x[1], reverse=True)
        
        print("Top 3 by Embedding:")
        for i, (doc, score) in enumerate(emb_results[:3]):
            status = "✅ (NEEDLE)" if doc["title"] == haystack[needle_index]["title"] else "❌"
            print(f"{i+1}. {doc['title']} - Score: {score:.4f} {status}")

    except Exception as e:
        print(f"❌ Embedding Step Failed: {e}")
        return

    # 4. Step 2: Reranking (ecnu-rerank)
    print(f"\n[Step 2] Reranking ({ECNU_RERANK_MODEL_NAME})...")
    rerank_url = f"{base_url}/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Use all docs for rerank to see if it can pick the needle
    payload = {
        "model": ECNU_RERANK_MODEL_NAME,
        "query": query,
        "documents": [doc["content"] for doc in haystack],
        "top_n": 5,
        "return_documents": False
    }
    
    try:
        response = requests.post(rerank_url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            rerank_data = response.json()
            print("Top 3 by Rerank:")
            for i, res in enumerate(rerank_data.get('results', [])[:3]):
                doc_idx = res['index']
                doc_title = haystack[doc_idx]['title']
                score = res['relevance_score']
                status = "✅ (NEEDLE)" if doc_idx == needle_index else "❌"
                print(f"{i+1}. {doc_title} - Score: {score:.4f} {status}")
        else:
            print(f"❌ Rerank Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Rerank Error: {e}")

if __name__ == "__main__":
    test_ecnu_models_niah()
