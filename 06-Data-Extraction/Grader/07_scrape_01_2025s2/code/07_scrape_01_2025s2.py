from bs4 import BeautifulSoup

# Constants
BUDDHIST_DAY_COLUMN_NAME = "div.bud-day-col"
DAY_PREFIX = "วัน"
DAY_NAMES = (
    "วันจันทร์",
    "วันอังคาร",
    "วันพุธ",
    "วันพฤหัสบดี",
    "วันศุกร์",
    "วันเสาร์",
    "วันอาทิตย์",
)
TARGET_DAY = "วันวิสาขบูชา"


# Utility Functions
def get_buddhist_day_columns(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "lxml")
    return soup.select(BUDDHIST_DAY_COLUMN_NAME)


def get_day_idx(text):
    if not text.startswith(DAY_PREFIX):
        return -1

    for idx, day_name in enumerate(DAY_NAMES):
        if text.startswith(day_name):
            return idx
    return -1


# Assignment Tasks
def Q1(file_path):
    buddhist_day_count = [0, 0, 0, 0, 0, 0, 0]

    for item in get_buddhist_day_columns(file_path):
        # Skip if current item has no text
        if not isinstance(item.text, str):
            continue

        # Skip if the current item is not a date
        idx = get_day_idx(item.text.strip())
        if idx == -1:
            continue

        # Update the counting list
        buddhist_day_count[idx] += 1

    return buddhist_day_count


def Q2(file_path, target_day=TARGET_DAY):
    # Initialize the variable to store the holiday date
    holiday_date = ""

    # Get text for all buddhist day column tag field
    buddhist_day_texts = [item.text for item in get_buddhist_day_columns(file_path)]

    for idx in range(0, len(buddhist_day_texts), 3):
        # Get date and holiday info field
        date = buddhist_day_texts[idx]
        holiday_info = buddhist_day_texts[idx + 2]

        # If the holiday info are empty, skip it
        if not holiday_info:
            continue

        # If the current holiday is not the target, skip it
        if target_day not in holiday_info.strip():
            continue

        # Update the holiday date variable field
        holiday_date = date
        break

    return holiday_date


if __name__ == "__main__":
    exec(input().strip())
