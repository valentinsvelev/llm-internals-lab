import shutil


def pytest_sessionstart(session):
    try:
        import torch
    except ImportError:          # the explorer tests do not need PyTorch
        return
    torch.set_num_threads(1)


def launch_chromium(playwright):
    """A headless Chromium for the browser test, or None.

    Tries Playwright's own build (`python -m playwright install chromium`), then an installed
    Google Chrome (found in its standard location on macOS, Windows and Linux), then chromium or
    google-chrome on the PATH."""
    attempts = [{}, {'channel': 'chrome'}]
    for name in ('chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'):
        if path := shutil.which(name):
            attempts.append({'executable_path': path})
    for options in attempts:
        try:
            return playwright.chromium.launch(headless=True, args=['--no-sandbox'], **options)
        except Exception:  # not installed, or not runnable here: try the next one
            continue
    return None
