import json
import re
from pathlib import Path

# Load the original data
with open("data.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)

converted_data = []

for entry in original_data:
    based_on = entry.get("basedOn", "").strip()

    technique = ""
    base_model = based_on

    match = re.match(r"^(.*?)(?:\s*\((SFT|RL)\))?$", based_on)
    if match:
        base_model = match.group(1).strip()
        if match.group(2):
            technique = match.group(2)

    # Create the new entry
    new_entry = {
        "name": entry.get("name", ""),
        "organization": entry.get("organization", ""),
        "baseModel": base_model,
        "technique": technique,
        "paperLink": entry.get("paperLink", ""),
        "paperText": entry.get("paperText", ""),
        "aime24": entry.get("aime24", ""),
        "aime25": entry.get("aime25", ""),
        "amc23": entry.get("amc23", ""),
        "math500": entry.get("math500", ""),
        "minerva": entry.get("minerva", ""),
        "olympiad": entry.get("olympiad", "")
    }

    converted_data.append(new_entry)

# Save the new data
with open("data_converted.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print("✅ Converted JSON saved to data_converted.json")
