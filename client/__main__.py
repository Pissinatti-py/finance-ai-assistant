"""Tiny streaming chat client for the WebSocket /chat endpoint.

Usage::

    python -m client                      # connects to ws://localhost:8000
    python -m client --url ws://host:8000 --api-key SECRET

Reads ANTHROPIC-free config from the environment: the URL defaults to
``CHAT_URL`` (or localhost) and the key to ``AI_API_KEY``. Type a question and
watch the answer stream in; the session id is kept so the conversation has
memory. Ctrl-D or "sair" to quit.
"""

import argparse
import asyncio
import json
import os

import websockets
from rich.console import Console

console = Console()


async def chat(url: str, api_key: str) -> None:
    """
    Open one WebSocket session and run the interactive prompt loop.

    :param url: Base server URL, e.g. ``ws://localhost:8000``.
    :type url: str
    :param api_key: The shared secret sent as ``?api_key=`` for the handshake.
    :type api_key: str
    :return: Nothing.
    :rtype: None
    """
    uri = f"{url.rstrip('/')}/chat?api_key={api_key}"
    session_id: str | None = None
    try:
        async with websockets.connect(uri) as ws:
            console.print("[bold green]Conectado.[/] Pergunte algo (Ctrl-D ou 'sair' para encerrar).\n")
            while True:
                try:
                    question = console.input("[bold cyan]você ›[/] ").strip()
                except EOFError:
                    break
                if not question or question.lower() in {"sair", "exit", "quit"}:
                    break

                await ws.send(json.dumps({"question": question, "lang": "pt-br", "session_id": session_id}))
                console.print("[bold magenta]assistente ›[/] ", end="")
                async for raw in ws:
                    event = json.loads(raw)
                    kind = event.get("type")
                    if kind == "token":
                        console.print(event["content"], end="", soft_wrap=True)
                    elif kind == "error":
                        console.print(f"\n[bold red]erro:[/] {event['detail']}")
                        break
                    elif kind == "done":
                        session_id = event.get("session_id") or session_id
                        if event.get("cached"):
                            console.print(event.get("answer", ""), end="")
                            console.print("  [dim](cache hit)[/]")
                        else:
                            console.print()  # newline after streamed answer
                        break
                console.print()
    except (OSError, websockets.exceptions.WebSocketException) as exc:
        console.print(f"[bold red]Falha de conexão:[/] {exc}")
        console.print("O servidor está rodando? Tente: [bold]docker compose up[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming chat client for finance-ai-assistant.")
    parser.add_argument("--url", default=os.getenv("CHAT_URL", "ws://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("AI_API_KEY", "change-me"))
    args = parser.parse_args()
    asyncio.run(chat(args.url, args.api_key))


if __name__ == "__main__":
    main()
