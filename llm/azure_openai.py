import os

from openai import AzureOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
import re
import json
import openai


def get_openai_client() -> AzureOpenAI:
    """
    Create Azure OpenAI client.

    Returns:
        Azure OpenAI client.
    """
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )


def _normalize_year_breakdowns(data: dict, year: int) -> dict:
    """
    Some model responses return a per-year breakdown for a field
    (e.g. {"2024": ..., "2023": ..., "2022": ...}) instead of a single
    value, despite prompt instructions asking for one year only.

    Collapse any such dict-valued field down to the value for the
    requested year, falling back to the most recent year present if
    the exact year key is missing.
    """
    normalized = dict(data)

    for key, value in data.items():
        if isinstance(value, dict) and value:
            if str(year) in value:
                normalized[key] = value[str(year)]
            elif year in value:
                normalized[key] = value[year]
            else:
                try:
                    latest_key = max(value.keys(), key=lambda k: int(k))
                    normalized[key] = value[latest_key]
                except (ValueError, TypeError):
                    normalized[key] = next(iter(value.values()))

    return normalized


def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
    model: str | None = None,
    year: int | None = None
) -> BaseModel:
    """
    Generate structured output.

    Args:
        prompt: Input prompt.
        response_model: Pydantic response model.
        model: Azure OpenAI deployment name.
        year: Fiscal year being requested, used to collapse multi-year
            breakdowns the model may return despite instructions.

    Returns:
        Parsed response model.
    """
    # Read deployment name from environment when not provided; do not fallback to a hardcoded name
    model = model or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")


    client = get_openai_client()

    messages = [
        {
            "role": "system",
            "content": "You are an expert financial analyst."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Prefer structured parsing when supported by the deployment
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model
        )

        # Debug: show structured parsed output
        try:
            parsed = response.choices[0].message.parsed
            print("[debug] Structured parsed output:", parsed)
        except Exception:
            print("[debug] Structured response received but could not access parsed field")

        return response.choices[0].message.parsed

    except openai.BadRequestError as exc:
        # Fallback: some Azure deployments/models don't support structured outputs.
        # Request a JSON-only response and parse it with pydantic.
        msg = str(exc)
        if "response_format" in msg or "json_schema" in msg or "Structured Outputs" in msg:
            fallback = client.chat.completions.create(
                model=model,
                messages=messages
            )

            text = fallback.choices[0].message.content
            print("[debug] Fallback raw text response:\n", text)

            # Try to extract the first JSON object from the model output
            match = re.search(r"\{.*\}", text, re.S)
            json_text = match.group(0) if match else text

            try:
                # Handle explicit 'null' responses: return an empty model instance
                if isinstance(json_text, str) and json_text.strip() in ("null", "None", ""):
                    # Prefer pydantic v2 `model_construct` (no validation) to build an empty model
                    if hasattr(response_model, "model_construct"):
                        return response_model.model_construct()
                    # Fallback: attempt to validate empty dict (may fail if required fields exist)
                    try:
                        return response_model.model_validate({}) if hasattr(response_model, "model_validate") else response_model.parse_obj({})
                    except Exception:
                        raise RuntimeError("Model returned null and cannot construct an empty instance; please check the model schema or use a deployment that supports structured outputs.")

                # Parse to a plain dict first so we can normalize shapes the
                # model may not have followed exactly (e.g. multi-year
                # breakdowns), then validate the normalized dict.
                data = json.loads(json_text)

                if year is not None and isinstance(data, dict):
                    data = _normalize_year_breakdowns(data, year)

                if hasattr(response_model, "model_validate"):
                    parsed = response_model.model_validate(data)
                else:
                    parsed = response_model.parse_obj(data)

                return parsed
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON fallback response: {e}\nRaw output:\n{text}") from e

        # Re-raise if it's a different bad request
        raise
