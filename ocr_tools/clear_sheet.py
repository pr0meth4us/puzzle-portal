import requests
from google.oauth2 import service_account
import google.auth.transport.requests

# 1. Get credentials via Service Account
key_path = "/Users/nicksng/code/puzzle-portal/ocr_tools/service_account.json"
credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
request = google.auth.transport.requests.Request()
credentials.refresh(request)

# 2. Clear range A99:H300
spreadsheet_id = "1DHZnLwUpQ30Ruoz-gSK8PSin2nfcqsjJZogZGCD2SxQ"
range_name = "'Actual Attendance MDK'!A99:H300"

url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}:clear"
headers = {
    "Authorization": f"Bearer {credentials.token}",
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers)
print("Response status code:", response.status_code)
if response.status_code == 200:
    print("Successfully cleared the appended rows from the sheet.")
else:
    print("Error clearing sheet:", response.text)
