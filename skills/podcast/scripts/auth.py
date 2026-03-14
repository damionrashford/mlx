# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "notebooklm>=0.3.0",
# ]
# ///
"""
NotebookLM authentication helper — check status and guide login.
"""

import argparse
import asyncio
import json
import sys


async def check_auth(test_connection: bool = False, output_json: bool = False) -> dict:
    """Check if NotebookLM authentication is configured."""
    from notebooklm import NotebookLMClient
    from notebooklm.paths import get_storage_state_path

    result = {
        "authenticated": False,
        "storage_state_exists": False,
        "storage_path": str(get_storage_state_path()),
        "connection_tested": False,
        "connection_ok": False,
        "error": None,
    }

    storage_path = get_storage_state_path()
    if storage_path.exists():
        result["storage_state_exists"] = True
        try:
            with open(storage_path) as f:
                state = json.load(f)
            cookies = state.get("cookies", [])
            cookie_names = {c["name"] for c in cookies}
            required = {"SID", "HSID"}
            if required.issubset(cookie_names):
                result["authenticated"] = True
            else:
                missing = required - cookie_names
                result["error"] = f"Missing required cookies: {', '.join(missing)}"
        except (json.JSONDecodeError, KeyError) as e:
            result["error"] = f"Invalid storage state: {e}"
    else:
        result["error"] = "No storage state found. Run: python3 auth.py login"

    if test_connection and result["authenticated"]:
        result["connection_tested"] = True
        try:
            async with await NotebookLMClient.from_storage() as client:
                notebooks = await client.notebooks.list()
                result["connection_ok"] = True
                result["notebook_count"] = len(notebooks)
        except Exception as e:
            result["connection_ok"] = False
            result["error"] = f"Connection test failed: {e}"

    return result


async def do_login(browser: str = "chromium") -> dict:
    """Run interactive browser login."""
    try:
        from notebooklm.auth import login as nlm_login
    except ImportError:
        return {
            "success": False,
            "error": (
                "Playwright is required for login. Install with:\n"
                "  pip install 'notebooklm[browser]'\n"
                "  playwright install chromium"
            ),
        }

    try:
        await nlm_login(browser=browser)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="NotebookLM authentication: check status or login via browser.",
        epilog=(
            "Examples:\n"
            "  %(prog)s check              # Check if authenticated\n"
            "  %(prog)s check --test       # Check + test API connection\n"
            "  %(prog)s check --json       # JSON output\n"
            "  %(prog)s login              # Open browser for Google login\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check_p = sub.add_parser("check", help="Check authentication status")
    check_p.add_argument("--test", action="store_true", help="Test API connection")
    check_p.add_argument("--json", action="store_true", help="Output as JSON")

    login_p = sub.add_parser("login", help="Login via browser (requires playwright)")
    login_p.add_argument(
        "--browser",
        default="chromium",
        choices=["chromium", "firefox", "webkit"],
        help="Browser to use (default: chromium)",
    )

    args = parser.parse_args()

    if args.command == "check":
        result = asyncio.run(check_auth(test_connection=args.test, output_json=args.json))
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["authenticated"]:
                print(f"Authenticated: yes")
                print(f"Storage: {result['storage_path']}")
                if result["connection_tested"]:
                    if result["connection_ok"]:
                        print(f"Connection: ok ({result.get('notebook_count', '?')} notebooks)")
                    else:
                        print(f"Connection: failed — {result['error']}")
                        sys.exit(1)
            else:
                print(f"Authenticated: no")
                print(f"Error: {result['error']}")
                print(f"\nTo authenticate, run:")
                print(f"  python3 {sys.argv[0]} login")
                sys.exit(1)

    elif args.command == "login":
        result = asyncio.run(do_login(browser=args.browser))
        if result["success"]:
            print("Login successful. Credentials saved.")
        else:
            print(f"Login failed: {result['error']}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
