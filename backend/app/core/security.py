import base64
import hashlib
import hmac
import json
import time

from app.core.config import SECRET_KEY


def hash_password(password: str) -> str:
    salt = hashlib.sha256(
        f"{SECRET_KEY}:{password}".encode()
    ).hexdigest()

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100_000,
    ).hex()

    return f"{salt}${password_hash}"


def verify_password(password: str, stored_password: str) -> bool:
    try:
        salt, stored_hash = stored_password.split("$", 1)

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            100_000,
        ).hex()

        return hmac.compare_digest(password_hash, stored_hash)

    except ValueError:
        return False


def create_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": int(time.time()) + 60 * 60 * 24,
    }

    payload_bytes = json.dumps(
        payload,
        separators=(",", ":")
    ).encode()

    payload_encoded = base64.urlsafe_b64encode(
        payload_bytes
    ).decode()

    signature = hmac.new(
        SECRET_KEY.encode(),
        payload_encoded.encode(),
        hashlib.sha256,
    ).hexdigest()

    return f"{payload_encoded}.{signature}"


def verify_token(token: str) -> dict | None:
    try:
        payload_encoded, signature = token.split(".", 1)

        expected_signature = hmac.new(
            SECRET_KEY.encode(),
            payload_encoded.encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(
            signature,
            expected_signature
        ):
            return None

        payload = json.loads(
            base64.urlsafe_b64decode(
                payload_encoded.encode()
            )
        )

        if payload["exp"] < int(time.time()):
            return None

        return payload

    except Exception:
        return None