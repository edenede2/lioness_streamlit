"""Replace the ignored uploader OAuth token without attempting an expired refresh.

Run on the SSH server with port 8765 forwarded from the browser's machine.
No Drive content is read or uploaded by this command.
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

from upload_prediction_data import DRIVE_SCOPE, _write_oauth_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oauth-client", type=Path, required=True)
    parser.add_argument("--oauth-token", type=Path, required=True)
    parser.add_argument("--backup-directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    os.umask(0o077)
    flow = InstalledAppFlow.from_client_secrets_file(str(args.oauth_client.resolve()), [DRIVE_SCOPE])
    try:
        credentials = flow.run_local_server(
            host="127.0.0.1", port=args.port, open_browser=False,
            timeout_seconds=args.timeout, prompt="consent", access_type="offline",
            authorization_prompt_message="Authorize the existing Drive uploader:\n{url}",
            success_message="Drive authorization completed. You may close this tab.",
        )
        if not credentials.valid or not credentials.refresh_token:
            raise RuntimeError("Authorization did not provide reusable credentials")
        token = args.oauth_token.resolve()
        if token.exists():
            backup = args.backup_directory.resolve()
            backup.mkdir(parents=True, exist_ok=True, mode=0o700)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            destination = backup / f"oauth-token-{stamp}.json"
            shutil.copyfile(token, destination)
            destination.chmod(0o600)
        _write_oauth_token(token, credentials)
        print("Authorization saved to the ignored local token. No files uploaded.", flush=True)
    except Exception as error:
        # OAuth exception details can contain the callback/code; never log them.
        print(f"Authorization did not finish ({type(error).__name__}); the old token is unchanged.", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
