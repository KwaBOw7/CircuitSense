
    
  
    response = _http(f"{BASE}/models?pageSize=200", key=key)
    names = [
        item["name"].split("/")[-1]
        for item in response.get("models", [])
        if "generateContent" in item.get("supportedGenerationMethods", [])
        and item.get("name")
    ]

    flash = [
        name for name in names
        if "flash" in name.lower()
        and not any(term in name.lower() for term in (
            "image", "tts", "live", "audio", "lite", "thinking", "exp", "preview"
        ))
    ]
    _model = next((candidate for candidate in PREFER if candidate in names), None)
    if not _model:
        # If Google's model list changes, prefer an available Flash model.
        _model = flash[0] if flash else (names[0] if names else None)
    if not _model:
        raise RuntimeError("No Gemini model with generateContent is available for this key")
    return _model


def _generate(parts, as_json):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    body = {"contents": [{"parts": parts}]}
    if as_json:
        body["generationConfig"] = {"response_mime_type": "application/json"}

    try:
        try:
            selected = _pick(key)
            result = _http(
                f"{BASE}/models/{selected}:generateContent",
                body,
                key=key,
            )
        except RuntimeError as exc:
            # A model can become unavailable after being listed. Rediscover once
            # on a 404, unless the operator explicitly pinned GEMINI_MODEL.
            if "HTTP 404" not in str(exc) or os.getenv("GEMINI_MODEL"):
                raise
            selected = _pick(key, refresh=True)
            result = _http(
                f"{BASE}/models/{selected}:generateContent",
                body,
                key=key,
            )

        output = result["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(output) if as_json else output
    except RuntimeError as exc:
        raise RuntimeError(f"Gemini call failed ({_model or 'model discovery'}): {exc}") from exc
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Gemini returned an unexpected response: {type(exc).__name__}"
        ) from exc


def generate(text, as_json=False):
    """Generate a text response, optionally requiring a JSON response."""
    return _generate([{"text": text}], as_json)


def resolve(png: bytes):
    """Identify one unrecognised symbol crop or return None when unavailable."""
    if not os.getenv("GEMINI_API_KEY"):
        return None
    try:
        result = _generate(
            [
                {"text": PROMPT},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(png).decode("ascii"),
                    }
                },
            ],
            True,
        )
    except RuntimeError:
        # Recognition must remain functional without the optional AI fallback.
        return None
    return result if isinstance(result, dict) and result.get("kind") in KINDS else None


def generate_image(text, png: bytes, as_json=False):
    """Generate a response using text plus an image, such as a ladder diagram."""
    return _generate(
        [
            {"text": text},
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(png).decode("ascii"),
                }
            },
        ],
        as_json,
    )

