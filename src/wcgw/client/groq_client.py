import base64
import json
import mimetypes
from pathlib import Path
import sys
import traceback
from typing import Callable, DefaultDict, Optional, cast
import groq
from groq.types.chat import ChatCompletion
import rich
import petname  # type: ignore[import-untyped]
from typer import Typer
import uuid
import signal
from contextlib import contextmanager

from ..types_ import (
    BashCommand,
    BashInteraction,
    CreateFileNew,
    FileEdit,
    ReadImage,
    ReadFile,
    ResetShell,
)

from .common import Models, discard_input
from .common import CostData, History
from .tools import ImageData

from .tools import (
    DoneFlag,
    get_tool_output,
    which_tool,
)
import tiktoken

from urllib import parse
import subprocess
import os
import tempfile

import toml
from pydantic import BaseModel
from dotenv import load_dotenv


class Config(BaseModel):
    model: Models
    cost_limit: float
    cost_file: dict[Models, CostData]
    cost_unit: str = "$"


def text_from_editor(console: rich.console.Console) -> str:
    # First consume all the input till now
    discard_input()
    console.print("\n---------------------------------------\n# User message")
    data = input()
    if data:
        return data
    editor = os.environ.get("EDITOR", "vim")
    with tempfile.NamedTemporaryFile(suffix=".tmp") as tf:
        subprocess.run([editor, tf.name], check=True)
        with open(tf.name, "r") as f:
            data = f.read()
            console.print(data)
            return data


def save_history(history: History, session_id: str) -> None:
    myid = str(history[1]["content"]).replace("/", "_").replace(" ", "_").lower()[:60]
    myid += "_" + session_id
    myid = myid + ".json"

    mypath = Path(".wcgw") / myid
    mypath.parent.mkdir(parents=True, exist_ok=True)
    with open(mypath, "w") as f:
        json.dump(history, f, indent=3)


def parse_user_message_special(msg: str) -> dict:
    # Search for lines starting with `%` and treat them as special commands
    parts = []
    for line in msg.split("\n"):
        if line.startswith("%"):
            args = line[1:].strip().split(" ")
            command = args[0]
            assert command == "image"
            image_path = " ".join(args[1:])
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                image_type = mimetypes.guess_type(image_path)[0]
                dataurl = f"data:{image_type};base64,{image_b64}"
            parts.append(
                {"type": "image_url", "image_url": {"url": dataurl, "detail": "auto"}}
            )
        else:
            if len(parts) > 0 and parts[-1]["type"] == "text":
                parts[-1]["text"] += "\n" + line
            else:
                parts.append({"type": "text", "text": line})
    return {"role": "user", "content": parts}


app = Typer(pretty_exceptions_show_locals=False)


@contextmanager
def timeout(time):
    def raise_timeout(signum, frame):
        raise TimeoutError(f"Operation timed out after {time} seconds")
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)
    try:
        yield
    finally:
        signal.alarm(0)


@app.command()
def loop(
    first_message: Optional[str] = None,
    limit: Optional[float] = None,
    resume: Optional[str] = None,
    computer_use: bool = False,
) -> tuple[str, float]:
    load_dotenv()
    print("Debug: Starting Groq client...")

    session_id = str(uuid.uuid4())[:6]
    print(f"Debug: Session ID: {session_id}")

    history: History = []
    waiting_for_assistant = False
    if resume:
        print("Debug: Resuming from existing session...")
        if resume == "latest":
            resume_path = sorted(Path(".wcgw").iterdir(), key=os.path.getmtime)[-1]
        else:
            resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"File {resume} not found")
        with resume_path.open() as f:
            history = json.load(f)
        if len(history) <= 2:
            raise ValueError("Invalid history file")
        first_message = ""
        waiting_for_assistant = history[-1]["role"] != "assistant"
    else:
        print("Debug: Initializing new session...")
        # Initialize with system message
        system_message = f"""You're an expert software engineer with shell and code knowledge.

Instructions:
    - You should use the provided bash execution, reading and writing file tools to complete objective.
    - First understand about the project by getting the folder structure (ignoring .git, node_modules, venv, etc.)
    - Always read relevant files before editing.
    - Do not provide code snippets unless asked by the user, instead directly add/edit the code.
    - Do not install new tools/packages before ensuring no such tools/package or an alternative already exists.

System information:
    - System: {os.uname().sysname}
    - Machine: {os.uname().machine}
    - Current directory: {os.getcwd()}
"""
        history = [{"role": "system", "content": system_message}]

    print("Debug: Setting up configuration...")
    config = Config(
        model=cast(Models, os.getenv("GROQ_MODEL", "llama3-groq-70b-8192-tool-use-preview").lower()),
        cost_limit=0.1,
        cost_unit="$",
        cost_file={
            "llama3-groq-70b-8192-tool-use-preview": CostData(
                cost_per_1m_input_tokens=0.20,  # Using LLaMA2 pricing as base
                cost_per_1m_output_tokens=0.20,
            ),
            "mixtral-8x7b-32768": CostData(
                cost_per_1m_input_tokens=0.27,
                cost_per_1m_output_tokens=0.27,
            ),
            "llama2-70b-4096": CostData(
                cost_per_1m_input_tokens=0.20,
                cost_per_1m_output_tokens=0.20,
            ),
        },
    )

    if limit is not None:
        config.cost_limit = limit
    limit = config.cost_limit
    print(f"Debug: Using model {config.model} with cost limit {limit}")

    print("Debug: Setting up tokenizer...")
    enc = tiktoken.encoding_for_model("gpt-4")

    print("Debug: Setting up tools...")
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool.__name__,
                "description": tool.__doc__ or "",
                "parameters": tool.model_json_schema(),
            },
        }
        for tool in [
            BashCommand,
            BashInteraction,
            ReadFile,
            CreateFileNew,
            FileEdit,
            ReadImage,
            ResetShell,
        ]
    ]

    print("Debug: Initializing Groq client...")
    client = groq.Client()

    print("Debug: Setting up console...")
    cost: float = 0
    input_toks = 0
    output_toks = 0
    system_console = rich.console.Console(style="blue", highlight=False, markup=False)
    error_console = rich.console.Console(style="red", highlight=False, markup=False)
    user_console = rich.console.Console(
        style="bright_black", highlight=False, markup=False
    )
    assistant_console = rich.console.Console(
        style="white bold", highlight=False, markup=False
    )

    print("Debug: Starting main loop...")
    while True:
        if cost > limit:
            system_console.print(
                f"\nCost limit exceeded. Current cost: {cost}, input tokens: {input_toks}, output tokens: {output_toks}"
            )
            break

        if not waiting_for_assistant:
            if first_message:
                print("Debug: Using first message...")
                msg = first_message
                first_message = ""
                history.append({"role": "user", "content": msg})
            else:
                print("Debug: Getting user input...")
                msg = text_from_editor(user_console)
                history.append(parse_user_message_special(msg))
        else:
            waiting_for_assistant = False

        print("Debug: Calculating input cost...")
        input_tokens = sum(len(enc.encode(str(msg["content"]))) for msg in history)
        cost += input_tokens * config.cost_file[config.model].cost_per_1m_input_tokens / 1_000_000
        input_toks += input_tokens

        try:
            print("Debug: Creating chat completion...")
            with timeout(10):  # 10 second timeout for API operations
                completion = client.chat.completions.create(
                    messages=history,
                    model=config.model,
                    stream=True,
                    tools=tools,
                    max_tokens=1000,  # Limit response size
                    temperature=0.7,  # Add some randomness to prevent loops
                )

                print("Debug: Processing response...")
                system_console.print(
                    "\n---------------------------------------\n# Assistant response",
                    style="bold",
                )

                # Handle streaming response
                collected_message = None
                response_text = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response_text += content
                        assistant_console.print(content, end="")
                    if chunk.choices[0].delta.tool_calls:
                        # If we get tool calls, we need to collect them
                        if collected_message is None:
                            collected_message = chunk.choices[0].delta
                        else:
                            # Merge the deltas
                            if chunk.choices[0].delta.tool_calls:
                                if not collected_message.tool_calls:
                                    collected_message.tool_calls = []
                                collected_message.tool_calls.extend(chunk.choices[0].delta.tool_calls)
                            if chunk.choices[0].delta.content:
                                if collected_message.content:
                                    collected_message.content += chunk.choices[0].delta.content
                                else:
                                    collected_message.content = chunk.choices[0].delta.content

                # Use the collected message if we have tool calls, otherwise use the last chunk
                message = collected_message if collected_message and collected_message.tool_calls else chunk.choices[0].delta

                # Check for potential infinite loops
                if len(history) > 10:  # If we have more than 10 exchanges
                    last_few_responses = [msg for msg in history[-6:] if msg["role"] == "assistant"]
                    if len(last_few_responses) >= 5:  # Check last 5 assistant responses
                        # If they're all similar (same tool calls with same args)
                        if all(
                            "tool_calls" in resp
                            and resp["tool_calls"] == last_few_responses[0]["tool_calls"]
                            for resp in last_few_responses[1:]
                        ):
                            error_console.print("\nDetected potential infinite loop, stopping...")
                            break

                if message.tool_calls:
                    tool_calls = message.tool_calls
                    item = {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "arguments": tool_call.function.arguments,
                                    "name": tool_call.function.name,
                                },
                            }
                            for tool_call in tool_calls
                        ],
                    }

                    output_tokens = len(enc.encode(str(item)))
                    cost += output_tokens * config.cost_file[config.model].cost_per_1m_output_tokens / 1_000_000
                    output_toks += output_tokens

                    system_console.print(
                        f"\n---------------------------------------\n# Assistant invoked tools: {[tool_call.function.name for tool_call in tool_calls]}"
                    )
                    system_console.print(f"\nTotal cost: {config.cost_unit}{cost:.3f}")

                    _histories = [item]

                    for tool_call in tool_calls:
                        try:
                            output_or_dones, cost_ = get_tool_output(
                                json.loads(tool_call.function.arguments),
                                enc,
                                limit - cost,
                                loop,
                                max_tokens=8000,
                            )
                            output_or_done = output_or_dones[0]
                        except Exception as e:
                            output_or_done = f"GOT EXCEPTION while calling tool. Error: {e}"
                            tb = traceback.format_exc()
                            error_console.print(output_or_done + "\n" + tb)
                            cost_ = 0

                        cost += cost_
                        system_console.print(f"\nTotal cost: {config.cost_unit}{cost:.3f}")

                        if isinstance(output_or_done, DoneFlag):
                            system_console.print(
                                f"\n# Task marked done, with output {output_or_done.task_output}",
                            )
                            system_console.print(f"\nTotal cost: {config.cost_unit}{cost:.3f}")
                            return output_or_done.task_output, cost

                        if isinstance(output_or_done, ImageData):
                            randomId = petname.Generate(2, "-")
                            item = {
                                "role": "tool",
                                "content": f"Ask user for image id: {randomId}",
                                "tool_call_id": tool_call.id,
                            }
                        else:
                            item = {
                                "role": "tool",
                                "content": str(output_or_done),
                                "tool_call_id": tool_call.id,
                            }

                        output_tokens = len(enc.encode(str(item)))
                        cost += output_tokens * config.cost_file[config.model].cost_per_1m_output_tokens / 1_000_000
                        output_toks += output_tokens

                        _histories.append(item)
                    waiting_for_assistant = True
                else:
                    content = message.content
                    assistant_console.print(content)
                    item = {
                        "role": "assistant",
                        "content": content,
                    }
                    output_tokens = len(enc.encode(str(item)))
                    cost += output_tokens * config.cost_file[config.model].cost_per_1m_output_tokens / 1_000_000
                    output_toks += output_tokens

                    system_console.print(f"\nTotal cost: {config.cost_unit}{cost:.3f}")
                    _histories = [item]

                history.extend(_histories)
                save_history(history, session_id)

                if first_message:
                    break

        except TimeoutError as e:
            error_console.print(f"Error: {e}")
            break
        except Exception as e:
            tb = traceback.format_exc()
            error_console.print(f"Error: {e}\n{tb}")
            break

    return "", cost


if __name__ == "__main__":
    app() 