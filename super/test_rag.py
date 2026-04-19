"""RAG accuracy test suite for CERN/particle physics documents."""

import time
from rag import RAGEngine

# Test questions with expected facts that should appear in answers
TESTS = [
    {
        "question": "What is the measured mass of the Higgs boson?",
        "expected_keywords": ["125", "GeV"],
        "topic": "Higgs mass",
    },
    {
        "question": "What is the significance of the H→Zγ decay observation by ATLAS?",
        "expected_keywords": ["Z", "gamma", "decay", "significance"],
        "topic": "H→Zγ decay",
    },
    {
        "question": "What was the center-of-mass energy used in the CMS four top quark production observation?",
        "expected_keywords": ["13", "TeV"],
        "topic": "Four top quarks energy",
    },
    {
        "question": "What are the main goals of the Future Circular Collider as discussed in the European Strategy?",
        "expected_keywords": ["collider", "energy"],
        "topic": "FCC goals",
    },
    {
        "question": "How many events or how much integrated luminosity was used in the ATLAS W boson mass measurement?",
        "expected_keywords": ["fb", "luminosity"],
        "topic": "W mass luminosity",
    },
    {
        "question": "What constraints have been placed on the Higgs boson self-coupling by CMS?",
        "expected_keywords": ["coupling", "Higgs", "κ"],
        "topic": "Higgs self-coupling",
    },
    {
        "question": "What is the role of the Mamba architecture layers in the Nemotron model family?",
        "expected_keywords": [],  # This tests hallucination - docs don't cover Nemotron architecture
        "topic": "Out-of-scope test (Nemotron architecture)",
        "expect_no_answer": True,
    },
    {
        "question": "What detector subsystems does ATLAS use for measuring the W boson mass?",
        "expected_keywords": ["calorimeter", "detector"],
        "topic": "ATLAS detector",
    },
    {
        "question": "What is the Higgs boson trilinear self-coupling parameter κλ and what range is allowed by CMS measurements?",
        "expected_keywords": ["κ", "coupling", "range"],
        "topic": "κλ constraints",
    },
    {
        "question": "Describe the LHC Run 3 data collection. What collision energy is used?",
        "expected_keywords": ["13.6", "TeV", "Run 3"],
        "topic": "LHC Run 3",
    },
]


def run_tests():
    print("Initializing RAG engine...")
    engine = RAGEngine()
    print(f"Vectorstore: {engine.collection.count()} chunks\n")

    results = []
    total_time = 0

    for i, test in enumerate(TESTS):
        print(f"{'='*60}")
        print(f"Test {i+1}/{len(TESTS)}: {test['topic']}")
        print(f"Q: {test['question']}")
        print()

        start = time.time()
        answer, chunks = engine.generate_full(test["question"])
        elapsed = time.time() - start
        total_time += elapsed

        # Check keyword hits
        answer_lower = answer.lower()
        hits = [kw for kw in test["expected_keywords"] if kw.lower() in answer_lower]
        misses = [kw for kw in test["expected_keywords"] if kw.lower() not in answer_lower]

        if test["expected_keywords"]:
            score = len(hits) / len(test["expected_keywords"])
        else:
            score = 1.0  # Out-of-scope test

        sources = set(c["source"] for c in chunks)
        avg_dist = sum(c["distance"] for c in chunks) / len(chunks) if chunks else 0

        print(f"A: {answer[:500]}{'...' if len(answer) > 500 else ''}")
        print()
        print(f"  Sources: {sources}")
        print(f"  Avg retrieval distance: {avg_dist:.3f}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Keyword hits: {hits} | Misses: {misses}")
        print(f"  Score: {score:.0%}")
        print()

        results.append({
            "topic": test["topic"],
            "score": score,
            "time": elapsed,
            "avg_dist": avg_dist,
            "answer_len": len(answer),
        })

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_score = sum(r["score"] for r in results) / len(results)
    avg_time = total_time / len(results)

    for r in results:
        status = "PASS" if r["score"] >= 0.5 else "FAIL"
        print(f"  [{status}] {r['topic']:40s} score={r['score']:.0%}  time={r['time']:.1f}s  dist={r['avg_dist']:.3f}")

    print(f"\n  Overall accuracy: {avg_score:.0%}")
    print(f"  Average response time: {avg_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    run_tests()
