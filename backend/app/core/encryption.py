"""
AES-256-GCM Encryption Service
================================
HIPAA §164.312(a)(2)(iv) — Encryption and Decryption (Addressable)
HIPAA §164.312(e)(2)(ii) — Encryption in Transit (Addressable)

All uploaded EHR documents are encrypted at rest using AES-256-GCM.
Each file gets a unique random salt so that even identical documents
produce completely different ciphertext.

Encrypted file layout  (.enc):
  [16 bytes  – random salt for HKDF key derivation]
  [12 bytes  – random GCM nonce                   ]
  [N  bytes  – AES-256-GCM ciphertext + 16-byte auth tag]
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    AESGCM = None
    SHA256 = None
    HKDF = None

from loguru import logger

# Context label keeps these keys domain-separated from any other
# usage of the same master key.
_KDF_INFO = b"ehr-ai-sidebar-file-encryption-v1"
_SALT_LEN = 16   # bytes for HKDF salt
_NONCE_LEN = 12  # bytes for GCM nonce (96-bit recommended)
_KEY_LEN = 32    # 256-bit AES key


class EncryptionService:
    """
    Wraps AES-256-GCM encryption/decryption for EHR document files.

    Usage:
        svc = EncryptionService(master_key_b64)
        enc_path = svc.encrypt_file(path)   # saves path.enc, deletes original
        plain_path = svc.decrypt_to_temp(enc_path)  # returns NamedTemp path
    """

    def __init__(self, master_key_b64: str) -> None:
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("Cryptography not available - encryption disabled")
        
        raw = base64.urlsafe_b64decode(master_key_b64 + "==")
        if len(raw) < 32:
            raise ValueError(
                "EHR_ENCRYPTION_KEY must be at least 32 bytes "
                "(use: python -c \"import secrets,base64; "
                "print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())\")"
            )
        self._master_key: bytes = raw[:32]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive a 256-bit file-specific AES key via HKDF-SHA256."""
        hkdf = HKDF(
            algorithm=SHA256(),
            length=_KEY_LEN,
            salt=salt,
            info=_KDF_INFO,
        )
        return hkdf.derive(self._master_key)

    # ── Public API ────────────────────────────────────────────────────────────

    def encrypt_bytes(self, plaintext: bytes) -> bytes:
        """
        Encrypt raw bytes.  Returns  salt || nonce || ciphertext+tag.
        """
        salt = os.urandom(_SALT_LEN)
        nonce = os.urandom(_NONCE_LEN)
        key = self._derive_key(salt)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        return salt + nonce + ciphertext

    def decrypt_bytes(self, blob: bytes) -> bytes:
        """
        Decrypt blob produced by encrypt_bytes().
        Raises InvalidTag if the data has been tampered with.
        """
        if len(blob) < _SALT_LEN + _NONCE_LEN + 16:
            raise ValueError("Encrypted blob is too short — corrupted data")
        salt = blob[:_SALT_LEN]
        nonce = blob[_SALT_LEN: _SALT_LEN + _NONCE_LEN]
        ciphertext = blob[_SALT_LEN + _NONCE_LEN:]
        key = self._derive_key(salt)
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, None)

    def encrypt_file(self, path: Path) -> Path:
        """
        Encrypt *path* in-place, writing *path*.enc and deleting the
        plaintext original.  Returns the .enc path.
        """
        plaintext = path.read_bytes()
        encrypted = self.encrypt_bytes(plaintext)
        enc_path = path.with_suffix(path.suffix + ".enc")
        enc_path.write_bytes(encrypted)
        path.unlink()  # delete plaintext — never leave PHI unencrypted on disk
        logger.debug(f"Encrypted {path.name} → {enc_path.name}")
        return enc_path

    def decrypt_to_temp(self, enc_path: Path) -> Path:
        """
        Decrypt *enc_path* into a NamedTemporaryFile and return its path.
        The caller is responsible for deleting the temp file after use.
        The original suffix (before .enc) is preserved so the processor
        knows the file type.
        """
        blob = enc_path.read_bytes()
        plaintext = self.decrypt_bytes(blob)

        # Recover original extension: foo.pdf.enc → foo.pdf
        original_suffix = enc_path.stem.rsplit(".", 1)[-1] if "." in enc_path.stem else ""
        suffix = f".{original_suffix}" if original_suffix else ""

        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
            dir=enc_path.parent,
        )
        tmp.write(plaintext)
        tmp.flush()
        tmp.close()
        logger.debug(f"Decrypted {enc_path.name} → tmp {Path(tmp.name).name}")
        return Path(tmp.name)


# ── Module-level singleton (initialised lazily from config) ───────────────────

_instance: Optional[EncryptionService] = None


def get_encryption_service() -> Optional[EncryptionService]:
    """
    Return the module singleton if encryption is enabled, else None.
    Call this lazily so the config is fully loaded before first use.
    """
    global _instance
    if _instance is not None:
        return _instance

    from app.config import settings  # deferred import avoids circular deps

    if not CRYPTOGRAPHY_AVAILABLE:
        logger.warning("Cryptography not available — encryption disabled")
        return None

    if not getattr(settings, "ENCRYPT_UPLOADS", False):
        return None

    key = getattr(settings, "EHR_ENCRYPTION_KEY", None)
    if not key:
        logger.warning(
            "ENCRYPT_UPLOADS=true but EHR_ENCRYPTION_KEY is not set — "
            "encryption disabled.  Set EHR_ENCRYPTION_KEY in .env to enable."
        )
        return None

    _instance = EncryptionService(key)
    logger.info("AES-256-GCM file encryption enabled")
    return _instance
