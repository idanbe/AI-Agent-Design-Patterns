from agent_patterns.p13_chain_of_verification.graph import build_graph, mock_llm, EXAMPLE_INPUT


def test_cove_verifies_claims_and_revises():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    assert out["baseline"] == "Go, Dart and Python."
    assert len(out["checks"]) == 3
    assert dict(out["verified"])["Was Python created at Google?"].startswith("No")
    assert out["final"].startswith("Go and Dart")
    assert llm.calls == 6


def test_verification_is_factored_from_the_draft():
    llm = mock_llm()
    out = build_graph(llm).invoke(EXAMPLE_INPUT)
    verify_prompts = [p for p in llm.prompts if p.startswith("Answer briefly")]
    assert len(verify_prompts) == 3 and all(out["baseline"] not in p for p in verify_prompts)


def test_simple_matches_graph():
    from agent_patterns.p13_chain_of_verification.simple import run   # local import: graph tests run before simple.py exists
    g_llm, s_llm = mock_llm(), mock_llm()
    g = build_graph(g_llm).invoke(EXAMPLE_INPUT)
    s = run(EXAMPLE_INPUT, s_llm)
    for key in ("baseline", "checks", "final"):
        assert s[key] == g[key], key
    assert sorted(s["verified"]) == sorted(g["verified"])
    assert s_llm.calls == g_llm.calls
