from datetime import datetime

def get_time():
    current_time = datetime.now().strftime("%I:%M %p")
    return f"The time is currently {current_time}"

def get_date():
    now = datetime.now()
    day_name = now.strftime("%A")
    month_name = now.strftime("%B")
    year = now.strftime("%Y")
    day_number = str(int(now.strftime("%d")))
    return f"Today is {day_name}, {month_name} {get_date_with_suffix(day_number)} {year}"

def get_date_with_suffix(date):
    suffix = ""
    if date in ("11", "12", "13"):
        suffix = "th"
    elif date[-1] == "1":
        suffix = "st"
    elif date[-1] == "2":
        suffix = "nd"
    elif date[-1] == "3":
        suffix = "rd"
    else:
        suffix = "th"
    return f"{date}{suffix}"