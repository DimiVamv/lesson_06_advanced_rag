def print_title(title: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def is_exit_command(text: str) -> bool:
    return text.strip().lower() in {"exit", "quit", "q", "bye", "goodbye"}