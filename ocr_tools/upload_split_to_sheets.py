import sys
import requests
from google.oauth2 import service_account
import google.auth.transport.requests

if len(sys.argv) < 2:
    print("Usage: python3 upload_split_to_sheets.py <SPREADSHEET_ID>")
    exit(1)

spreadsheet_id = sys.argv[1]

# 1. Parse Markdown file to split into June 13 and June 14 lists
md_path = "/Users/nicksng/code/puzzle-portal/ocr_tools/results/attendance_transcription.md"
with open(md_path, "r", encoding="utf-8") as f:
    content = f.read()

june13_rows = []
june14_rows = []
current_section = None

for line in content.split("\n"):
    line = line.strip()
    if "## 13-June-2026" in line:
        current_section = 13
        continue
    elif "## 14-June-2026" in line:
        current_section = 14
        continue
    
    if not line.startswith("|"):
        continue
    parts = [p.strip() for p in line.split("|")[1:-1]]
    if not parts or parts[0] == "ល.រ" or parts[0].startswith("---") or parts[0].startswith(":---"):
        continue
    
    no_val = parts[0]
    name = parts[1]
    gender = parts[2]
    role = parts[3]
    company = parts[4]
    phone = parts[5]
    
    # Translate Gender to Status (ស្រី or ប្រុស)
    if gender in ["F", "F ", "ស្រី"]:
        status = "ស្រី"
    elif gender in ["M", "M ", "ប្រុស"]:
        status = "ប្រុស"
    else:
        status = gender

    row_data = [
        None,         # Col A
        int(no_val) if no_val.isdigit() else no_val,  # Col B
        name,         # Col C
        None,         # Col D
        status,       # Col E
        role if role not in ["None", ""] else None,       # Col F
        company if company not in ["None", ""] else None, # Col G
        phone if phone not in ["None", ""] else None      # Col H
    ]
    
    if current_section == 13:
        june13_rows.append(row_data)
    elif current_section == 14:
        june14_rows.append(row_data)

print(f"Parsed {len(june13_rows)} rows for June 13 and {len(june14_rows)} rows for June 14.")

# 2. Get credentials via Service Account
key_path = "/Users/nicksng/code/puzzle-portal/ocr_tools/service_account.json"
credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
request = google.auth.transport.requests.Request()
credentials.refresh(request)

headers = {
    "Authorization": f"Bearer {credentials.token}",
    "Content-Type": "application/json"
}

# 3. Create tabs 13-June-2026 and 14-June-2026 if they do not exist
metadata_url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}"
meta_res = requests.get(metadata_url, headers=headers)
if meta_res.status_code != 200:
    print("Error getting spreadsheet metadata. Make sure the service account has Editor access.")
    print("Service Account Email: ocr-sheets-uploader@mac-project-7892.iam.gserviceaccount.com")
    print("Error response:", meta_res.text)
    exit(1)

meta_data = meta_res.json()
existing_titles = [sheet["properties"]["title"] for sheet in meta_data["sheets"]]

requests_body = []
if "13-June-2026" not in existing_titles:
    requests_body.append({"addSheet": {"properties": {"title": "13-June-2026"}}})
if "14-June-2026" not in existing_titles:
    requests_body.append({"addSheet": {"properties": {"title": "14-June-2026"}}})

if requests_body:
    batch_url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}:batchUpdate"
    batch_res = requests.post(batch_url, headers=headers, json={"requests": requests_body})
    if batch_res.status_code == 200:
        print("Created missing tabs successfully.")
    else:
        print("Error creating tabs:", batch_res.text)

# 4. Populate data and formulas
def generate_sheet_content(rows, total_count):
    content = [
        ['@dropdown', None, f'="Guests ("&SUM(D7:D150)&")"'],
        [],
        [None, None, 'Male', f'=COUNTIF($E$7:$E$150, "ប្រុស")', f'=D3/{total_count}'],
        [None, None, 'Female', f'=COUNTIF($E$7:$E$150, "ស្រី")', f'=D4/{total_count}'],
        [],
        [None, 'No', 'ឈ្មោះ/Name', 'Qty', 'Status', 'តួនាទី/Position', 'អង្គភាព/Organization', 'លេខទូរស័ព្ទ/Email']
    ]
    for r in rows:
        content.append(r)
    return content

june13_content = generate_sheet_content(june13_rows, len(june13_rows))
june14_content = generate_sheet_content(june14_rows, len(june14_rows))

# Write to 13-June-2026 sheet
write_url_13 = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/'13-June-2026'!A1?valueInputOption=USER_ENTERED"
write_res_13 = requests.put(write_url_13, headers=headers, json={"values": june13_content})
if write_res_13.status_code == 200:
    print("Populated June 13 tab successfully.")
else:
    print("Error writing June 13 tab:", write_res_13.text)

# Write to 14-June-2026 sheet
write_url_14 = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/'14-June-2026'!A1?valueInputOption=USER_ENTERED"
write_res_14 = requests.put(write_url_14, headers=headers, json={"values": june14_content})
if write_res_14.status_code == 200:
    print("Populated June 14 tab successfully.")
else:
    print("Error writing June 14 tab:", write_res_14.text)

# Delete default 'Sheet1' if it exists and we have the new tabs
meta_res = requests.get(metadata_url, headers=headers)
meta_data = meta_res.json()
sheets_info = {sheet["properties"]["title"]: sheet["properties"]["sheetId"] for sheet in meta_data["sheets"]}

if "Sheet1" in sheets_info and ("13-June-2026" in sheets_info or "14-June-2026" in sheets_info):
    delete_body = {"requests": [{"deleteSheet": {"sheetId": sheets_info["Sheet1"]}}]}
    delete_url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}:batchUpdate"
    requests.post(delete_url, headers=headers, json=delete_body)
    print("Removed default 'Sheet1'.")

print("\nDone! View the spreadsheet here:")
print(f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}")
