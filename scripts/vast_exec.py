"""Remote execution helper for the rented Vast.ai box.

Supports key or password authentication so either path works without local ssh config.

    python scripts/vast_exec.py --cmd "nvidia-smi"
    python scripts/vast_exec.py --put local.txt /root/local.txt
    python scripts/vast_exec.py --get /root/out.json outputs/out.json

Credentials come from environment variables so they are never written to disk:
    PA3_HOST, PA3_PORT, PA3_USER
    PA3_SSH_KEY (path)  or  PA3_SSH_PASSWORD
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import paramiko

HOST = os.environ.get("PA3_HOST", "172.219.88.150")
PORT = int(os.environ.get("PA3_PORT", "10277"))
USER = os.environ.get("PA3_USER", "root")


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    password = os.environ.get("PA3_SSH_PASSWORD")
    key_path = os.environ.get("PA3_SSH_KEY")
    kwargs = {"hostname": HOST, "port": PORT, "username": USER, "timeout": 30}
    if password:
        kwargs["password"] = password
        kwargs["look_for_keys"] = False
        kwargs["allow_agent"] = False
    elif key_path:
        kwargs["key_filename"] = key_path
    client.connect(**kwargs)
    return client


def run(client, command: str, timeout: float | None = None) -> int:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    for line in iter(stderr.readline, ""):
        sys.stderr.write(line)
        sys.stderr.flush()
    return stdout.channel.recv_exit_status()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd")
    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    ap.add_argument("--timeout", type=float, default=None)
    args = ap.parse_args()

    client = connect()
    try:
        if args.cmd:
            sys.exit(run(client, args.cmd, args.timeout))
        if args.put:
            sftp = client.open_sftp()
            sftp.put(args.put[0], args.put[1])
            print(f"uploaded {args.put[0]} -> {args.put[1]}")
        if args.get:
            Path(args.get[1]).parent.mkdir(parents=True, exist_ok=True)
            sftp = client.open_sftp()
            sftp.get(args.get[0], args.get[1])
            print(f"downloaded {args.get[0]} -> {args.get[1]}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
