
from enum import Enum
from datetime import datetime

class Color(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

def printc(color:Color, text):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"\n {color.value}{current_time} {text} {Color.RESET.value}")