import re
import google.auth
import google.auth.transport.requests
import requests
import json

from google.oauth2 import service_account
import google.auth.transport.requests

# 1. Get credentials via Service Account
try:
    key_path = "/Users/nicksng/code/puzzle-portal/ocr_tools/service_account.json"
    credentials = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
except Exception as e:
    print(f"Error getting service account credentials: {e}")
    exit(1)


# 2. Parse Markdown file
md_path = "/Users/nicksng/code/puzzle-portal/ocr_tools/results/attendance_transcription.md"
with open(md_path, "r", encoding="utf-8") as f:
    content = f.read()

rows_to_append = []
current_num = 93  # Start numbering from 93 (next row after 92 in the sheet)

for line in content.split("\n"):
    line = line.strip()
    if not line.startswith("|"):
        continue
    parts = [p.strip() for p in line.split("|")[1:-1]]
    if not parts or parts[0] == "ល.រ" or parts[0].startswith("---") or parts[0].startswith(":---"):
        continue
    
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
        current_num,  # Col B
        name,         # Col C
        None,         # Col D
        status,       # Col E
        role if role not in ["None", ""] else None,       # Col F
        company if company not in ["None", ""] else None, # Col G
        phone if phone not in ["None", ""] else None      # Col H
    ]
    rows_to_append.append(row_data)
    current_num += 1

print(f"Parsed {len(rows_to_append)} rows from Markdown.")

# 3. Call Google Sheets API to append
spreadsheet_id = "1DHZnLwUpQ30Ruoz-gSK8PSin2nfcqsjJZogZGCD2SxQ"
range_name = "'Actual Attendance MDK'!A99:H"

url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}:append?valueInputOption=USER_ENTERED"
headers = {
    "Authorization": f"Bearer {credentials.token}",
    "Content-Type": "application/json"
}
body = {
    "values": rows_to_append
}

response = requests.post(url, headers=headers, json=body)
print("Response status code:", response.status_code)
if response.status_code == 200:
    print("Success! Attendance data successfully written to the sheet.")
else:
    print("Error writing to sheet:", response.text)
