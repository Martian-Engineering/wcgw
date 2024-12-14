import importlib
from typing import Optional
from typer import Typer
import typer

from .openai_client import loop as openai_loop
from .anthropic_client import loop as claude_loop
from .groq_client import loop as groq_loop


app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def loop(
    claude: bool = typer.Option(False, "--claude", help="Use Claude model"),
    groq: bool = typer.Option(False, "--groq", help="Use Groq model"),
    first_message: Optional[str] = typer.Option(None, "--first-message", help="Initial message to send"),
    limit: Optional[float] = typer.Option(None, "--limit", help="Cost limit in dollars"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume from a previous session"),
    computer_use: bool = typer.Option(False, "--computer-use", help="Enable computer use features"),
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
) -> tuple[str, float]:
    if version:
        version_ = importlib.metadata.version("wcgw")
        print(f"wcgw version: {version_}")
        exit()
    if claude:
        return claude_loop(
            first_message=first_message,
            limit=limit,
            resume=resume,
            computer_use=computer_use,
        )
    elif groq:
        return groq_loop(
            first_message=first_message,
            limit=limit,
            resume=resume,
            computer_use=computer_use,
        )
    else:
        return openai_loop(
            first_message=first_message,
            limit=limit,
            resume=resume,
        )


if __name__ == "__main__":
    app()
