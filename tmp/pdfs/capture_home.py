from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1.5)
    page.goto('http://localhost:8501')
    page.get_by_role('button',name='Computer network คืออะไร?',exact=True).wait_for(timeout=120000)
    page.wait_for_timeout(1500)
    page.screenshot(path='tmp/pdfs/screenshots/01_home.png')
    browser.close()
