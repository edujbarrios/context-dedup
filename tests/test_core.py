def test_package_imports():
    import context_dedup

    assert context_dedup.__all__ == []
