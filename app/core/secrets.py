PLACEHOLDER_SECRET = "CHANGE_ME_IN_PRODUCTION_please_use_openssl_rand_hex_32"


def is_placeholder_secret(key: str | None) -> bool:
    if not key or not str(key).strip():
        return True
    k = str(key).strip()
    if k == PLACEHOLDER_SECRET:
        return True
    if k.upper().startswith("CHANGE_ME"):
        return True
    return len(k) < 16
