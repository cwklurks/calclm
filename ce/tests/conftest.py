def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long emulator runs (deselect with -m 'not slow')")
