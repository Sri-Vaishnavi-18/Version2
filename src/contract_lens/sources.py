from __future__ import annotations

import ipaddress
import socket
from io import BytesIO
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_POLICY_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5


def _validate_public_http_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a complete public HTTP or HTTPS URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not accepted.")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("Only the standard HTTP and HTTPS ports are accepted.")
    hostname = parsed.hostname.rstrip(".")
    if hostname.lower() == "localhost" or hostname.lower().endswith(".localhost"):
        raise ValueError("Localhost URLs are not accepted.")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, None)}
    except OSError as error:
        raise ValueError(f"Could not resolve the policy host: {error}") from error
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("The URL must resolve only to public IP addresses.")
    return url


def extract_pdf_text(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content))
    except (PdfReadError, EOFError, ValueError) as error:
        raise ValueError(f"Could not read the PDF: {error}") from error
    if reader.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported.")
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("No extractable text was found in the PDF.")
    return text


def fetch_policy_url(url: str) -> str:
    current_url = url
    headers = {"User-Agent": "AI-Contract-Lens/1.0 (policy text fetcher)"}
    for redirect_count in range(MAX_REDIRECTS + 1):
        _validate_public_http_url(current_url)
        try:
            response = requests.get(
                current_url,
                headers=headers,
                timeout=(5, 15),
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException as error:
            raise ValueError(f"Could not fetch the policy URL: {error}") from error
        with response:
            if response.is_redirect or response.is_permanent_redirect:
                if redirect_count == MAX_REDIRECTS:
                    raise ValueError("The URL redirected too many times.")
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("The policy URL returned a redirect without a destination.")
                current_url = urljoin(current_url, location)
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                raise ValueError(f"The policy server returned an HTTP error: {error}") from error
            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > MAX_POLICY_BYTES:
                raise ValueError("The policy URL is larger than the 5 MB limit.")
            content = bytearray()
            try:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    content.extend(chunk)
                    if len(content) > MAX_POLICY_BYTES:
                        raise ValueError("The policy URL is larger than the 5 MB limit.")
            except requests.RequestException as error:
                raise ValueError(f"Could not read the policy response: {error}") from error
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if content_type == "application/pdf" or current_url.lower().endswith(".pdf"):
                return extract_pdf_text(bytes(content))
            if content_type in {"text/plain", "text/html", "application/xhtml+xml"} or not content_type:
                encoding = response.encoding or "utf-8"
                text = bytes(content).decode(encoding, errors="replace")
                if "html" in content_type or "<html" in text[:1000].lower():
                    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
                return text.strip()
            raise ValueError(f"Unsupported policy content type: {content_type or 'unknown'}.")
    raise ValueError("Unable to retrieve the policy URL.")
