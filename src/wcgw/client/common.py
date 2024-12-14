import select
import sys
import termios
import tty
from typing import Literal
from pydantic import BaseModel


class CostData(BaseModel):
    cost_per_1m_input_tokens: float
    cost_per_1m_output_tokens: float


from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ParsedChatCompletionMessage,
)

History = list[ChatCompletionMessageParam]
Models = Literal[
    "mixtral-8x7b-32768",
    "llama2-70b-4096",
    "llama3-groq-70b-8192-tool-use-preview",
]


def discard_input() -> None:
    try:
        # Get the file descriptor for stdin
        fd = sys.stdin.fileno()

        # Save current terminal settings
        old_settings = termios.tcgetattr(fd)

        try:
            # Switch terminal to non-canonical mode where input is read immediately
            tty.setcbreak(fd)

            # Discard all input
            while True:
                # Check if there is input to be read
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.read(1)  # Read one character at a time to flush the input buffer
                else:
                    break
        finally:
            # Restore old terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (termios.error, ValueError) as e:
        # Handle the error gracefully
        print(f"Warning: Unable to discard input. Error: {e}")
