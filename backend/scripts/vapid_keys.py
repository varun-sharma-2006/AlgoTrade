"""Generate a VAPID key pair for Web Push: `python -m backend.scripts.vapid_keys`.

Put VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY in the server's environment (never commit the private key).
"""

import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def generate() -> tuple[str, str]:
    key = ec.generate_private_key(ec.SECP256R1())
    private = key.private_numbers().private_value.to_bytes(32, "big")
    public = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return b64(public), b64(private)


if __name__ == "__main__":
    public, private = generate()
    print(f"VAPID_PUBLIC_KEY={public}")
    print(f"VAPID_PRIVATE_KEY={private}")
