from rag import RAGSystem
import os

print("--- Testing RAG System ---")

# Ensure dummy docs exist
if not os.path.exists("data/docs/investment_strategy.md"):
    print("Error: Dummy docs not found in data/docs/")
    exit(1)

try:
    rag = RAGSystem()
    
    # Test 1: Retrieval
    query = "What is the recommended asset allocation for equities?"
    print(f"\nQuery: {query}")
    relevant = rag.retrieve(query)
    print(f"Retrieved {len(relevant)} chunks.")
    if len(relevant) > 0:
        print(f"Top Source: {relevant[0]['source']}")
        print(f"Snippet: {relevant[0]['text'][:100]}...")
    else:
        print("Unit Test Failed: No chunks retrieved.")

    # Test 2: Generation (Ollama)
    print("\nTesting Generation (this might take a few seconds)...")
    answer = rag.generate_answer(query)
    print(f"\nAnswer:\n{answer}")

    print("\n--- Test Complete ---")

except Exception as e:
    print(f"Test Failed with error: {e}")
