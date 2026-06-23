import requests
from google.oauth2 import service_account
import google.auth.transport.requests

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
        int(no_val) if no_val.isdigit() else no_val,  # Col B (No)
        name,         # Col C (ឈ្មោះ/Name)
        None,         # Col D (Qty)
        status,       # Col E (Status)
        role if role not in ["None", ""] else None,       # Col F (តួនាទី/Position)
        company if company not in ["None", ""] else None, # Col G (អង្គភាព/Organization)
        phone if phone not in ["None", ""] else None      # Col H (លេខទូរស័ព្ទ/Email)
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
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
)
request = google.auth.transport.requests.Request()
credentials.refresh(request)

# 3. Create a new Google Spreadsheet
headers = {
    "Authorization": f"Bearer {credentials.token}",
    "Content-Type": "application/json"
}

create_body = {
    "properties": {
        "title": "Attendance Report - 13 & 14 June 2026"
    },
    "sheets": [
        {"properties": {"title": "13-June-2026"}},
        {"properties": {"title": "14-June-2026"}}
    ]
}

create_url = "https://sheets.googleapis.com/v4/spreadsheets"
create_res = requests.post(create_url, headers=headers, json=create_body)
if create_res.status_code != 200:
    print("Error creating spreadsheet:", create_res.text)
    exit(1)

spreadsheet_data = create_res.json()
spreadsheet_id = spreadsheet_data["spreadsheetId"]
spreadsheet_url = spreadsheet_data["spreadsheetUrl"]
print(f"Created Spreadsheet: {spreadsheet_url}")

# 4. Share the Spreadsheet with the user's email
share_url = f"https://www.googleapis.com/drive/v3/files/{spreadsheet_id}/permissions?sendNotificationEmail=true"
share_body = {
    "role": "editor",
    "type": "user",
    "emailAddress": "phearaneron.soeung@gmail.com"
}
share_res = requests.post(share_url, headers=headers, json=share_body)
if share_res.status_code == 200:
    print("Successfully shared spreadsheet with phearaneron.soeung@gmail.com")
else:
    print("Error sharing spreadsheet:", share_res.text)

# 5. Populate June 13 sheet
def generate_sheet_content(rows, total_count):
    content = [
        # Row 1
        ['@dropdown', None, f'="Guests ("&SUM(D7:D150)&")"'],
        # Row 2
        [],
        # Row 3
        [None, None, 'Male', f'=COUNTIF($E$7:$E$150, "ប្រុស")', f'=D3/{total_count}'],
        # Row 4
        [None, None, 'Female', f'=COUNTIF($E$7:$E$150, "ស្រី")', f'=D4/{total_count}'],
        # Row 5
        [],
        # Row 6
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

print(f"\nFinal Spreadsheet Link: {spreadsheet_url}")
