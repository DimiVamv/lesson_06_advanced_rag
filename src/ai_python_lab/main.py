from ai_python_lab.claude_client import ClaudeClient
from ai_python_lab.utils import is_exit_command, print_title

def main():
    print_title("Welcome to the Claude Assistant Command Line Interface (CLI)!")
    
    claude_client = ClaudeClient()

    print("Type a question for Claude.")
    print("Type 'exit' to quit.")

    while True:
        prompt = input("You: ")
        if is_exit_command(prompt):
            print("\nGoodbye!")
            break

        try:
            response = claude_client.ask_claude(prompt, max_tokens=250)
            print(f"Claude's response:", response)
        except Exception as e:
            print(f"Error occurred:", e)

if __name__ == "__main__":
    main()
