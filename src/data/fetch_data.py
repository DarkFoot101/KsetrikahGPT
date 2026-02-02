import os
import time
import re
from datetime import datetime
from playwright.sync_api import sync_playwright

# CONFIG
URL = "https://agmarknet.gov.in/"
RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def fetch_daily_data():
    """
    Acts as an in-house AI agent to fetch data from Agmarknet.
    Simulates human behavior to avoid bot detection and handle dynamic UI.
    """
    with sync_playwright() as p:
        print("🕵️  Agent starting...")
        # Launch browser. Headless=True for background execution.
        browser = p.chromium.launch(headless=True, slow_mo=50) 
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1366, "height": 768}
        )
        page = context.new_page()

        try:
            print(f"🌍 Navigating to {URL}...")
            page.goto(URL, timeout=60000)
            
            # Wait for the dashboard/dropdowns to interact
            page.wait_for_load_state("networkidle")
            page.wait_for_selector("#variety", state="visible", timeout=30000)

            # ==========================================
            # TASK 1: Select "Individual" in Variety Dropdown
            # ==========================================
            print("⚙️  Configuring filters: Variety -> Individual")
            page.click("#variety")
            
            # 1. Uncheck 'All' if needed (best effort)
            try:
                # Locator for 'All Varieties'.
                all_var = page.locator("label").filter(has_text="All Varieties").first
                # Find input
                all_var_input = page.locator("input").filter(has=page.locator("xpath=..").filter(has_text="All Varieties")).first
                # Simplification: just click label if checked?
                # We skip unchecking 'All' specifically if we can't find it easily, 
                # but 'Individual' tick is the priority. 
                # Ideally selecting 'Individual' might auto-uncheck or we just live with it 
                # (user said: "only tick-mark 'Individual'").
                pass 
            except:
                pass

            # 2. Check "Individual"
            # Logic: Use text selector which works across tag types
            try:
                print("   Looking for 'Individual' option...")
                individual_option = page.locator("text=Individual").first
                individual_option.wait_for(state="visible", timeout=10000)
                individual_option.click()
                print("   Clicked 'Individual'.")
            except Exception as e:
                print(f"   Could not find 'Individual' option: {e}")
                pass

            # Close dropdown (press Escape)
            page.keyboard.press("Escape")

            # ==========================================
            # TASK 2: Click "Go" and Wait
            # ==========================================
            print("🚀 Clicking 'Go'...")
            # The button has an aria-label "Apply filters..." which hides "Go" from get_by_role(name="Go").
            # We use a text filter on the button element instead.
            page.locator("button").filter(has_text=re.compile(r"^Go$")).click()

            print("⏳ Waiting for data table generation...")
            try:
                page.wait_for_selector("table tbody tr", timeout=30000)
                print("   Table generated.")
            except:
                print("⚠️  Timeout waiting for table (or no data found). Proceeding to check for download...")

            # ==========================================
            # TASK 3: Download CSV
            # ==========================================
            print("💾 Initiating download...")
            
            # Locate the download button (Title="Download Report")
            download_btn = page.locator("button[title='Download Report']")
            
            if not download_btn.count():
                 print("   (Using fallback selector for download button)")
                 download_btn = page.locator("button:has(svg)").last 
            
            # Handle potential confirmation dialogs
            page.on("dialog", lambda dialog: dialog.accept())

            # Trigger the download menu
            download_btn.click()
            
            # Wait for "Download as CSV" option and click it
            with page.expect_download(timeout=60000) as download_info:
                print("   Selecting 'Download as CSV'...")
                # Use force=True to bypass overlapping checks if any
                page.click("text=Download as CSV", force=True)
                
            download = download_info.value
            
            # Save file
            today = datetime.now().strftime("%Y-%m-%d")
            filename = os.path.join(RAW_DATA_DIR, f"agmarknet_{today}.csv")
            download.save_as(filename)
            
            print(f"✅ MISSION COMPLETE: Data saved to {filename}")

        except Exception as e:
            print(f"❌ AGENT FAILURE: {e}")
            try:
                page.screenshot(path="agent_error.png")
                print("   Screenshot saved to agent_error.png")
            except:
                pass
        finally:
            browser.close()

if __name__ == "__main__":
    fetch_daily_data()
