#!/usr/bin/env python
"""Generate markdown API documentation from the FastAPI OpenAPI schema.

Usage:
    python scripts/generate_api_docs.py [--output docs/api.md]

The script imports the FastAPI app from server/main.py, extracts the OpenAPI
schema, and renders it as structured markdown grouped by tag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "src"))


def _schema_name(schema: Dict[str, Any]) -> str:
    """Extract a human-readable name from a $ref schema reference."""
    ref = schema.get("$ref", "")
    return ref.split("/")[-1] if ref else ""


def _format_schema(schema: Dict[str, Any], components: Dict[str, Any], depth: int = 0) -> str:
    """Format a JSON schema as a readable type description."""
    if "$ref" in schema:
        name = _schema_name(schema)
        # Don't recurse into referenced models — just show the name
        return f"`{name}`"
    schema_type = schema.get("type", "any")
    if schema_type == "array":
        items = schema.get("items", {})
        return f"array of {_format_schema(items, components, depth + 1)}"
    if schema_type == "object":
        props = schema.get("properties", {})
        if not props:
            return "object"
        if depth > 1:
            return "object"
        lines = ["object:"]
        for prop_name, prop_schema in props.items():
            required = prop_name in schema.get("required", [])
            req_marker = " *(required)*" if required else ""
            lines.append(f"  - `{prop_name}`{_req_marker}: {_format_schema(prop_schema, components, depth + 1)}")
        return "\n".join(lines)
    if schema_type == "string":
        enum = schema.get("enum")
        if enum:
            return f"string (one of: {', '.join(f'`{e}`' for e in enum)})"
        return "string"
    if schema_type == "integer":
        return "integer"
    if schema_type == "number":
        return "number"
    if schema_type == "boolean":
        return "boolean"
    return schema_type


def _format_parameters(parameters: List[Dict[str, Any]], components: Dict[str, Any]) -> str:
    """Format path/query parameters as a markdown table."""
    if not parameters:
        return "*No parameters*"
    lines = ["| Name | In | Type | Required | Description |", "|------|-----|------|----------|-------------|"]
    for param in parameters:
        name = param.get("name", "")
        location = param.get("in", "")
        required = "Yes" if param.get("required") else "No"
        desc = param.get("description", "").replace("\n", " ") or ""
        schema = param.get("schema", {})
        ptype = schema.get("type", "any")
        if schema.get("enum"):
            ptype = f"{ptype} ({', '.join(schema['enum'])})"
        lines.append(f"| `{name}` | {location} | {ptype} | {required} | {desc} |")
    return "\n".join(lines)


def _format_request_body(body: Dict[str, Any], components: Dict[str, Any]) -> str:
    """Format a request body schema."""
    if not body:
        return "*No request body*"
    content = body.get("content", {})
    json_schema = content.get("application/json", {}).get("schema", {})
    if not json_schema:
        return "*No request body*"
    required = " *(required)*" if body.get("required") else " *(optional)*"
    return f"```json\n{_format_schema(json_schema, components)}\n```{required}"


def _format_responses(responses: Dict[str, Any], components: Dict[str, Any]) -> str:
    """Format response schemas as a list."""
    if not responses:
        return "*No responses documented*"
    lines = []
    for status, response in sorted(responses.items()):
        desc = response.get("description", "")
        content = response.get("content", {})
        json_schema = content.get("application/json", {}).get("schema", {})
        schema_str = _format_schema(json_schema, components) if json_schema else "*no schema*"
        lines.append(f"- **{status}**: {desc} → {schema_str}")
    return "\n".join(lines)


def generate_docs(openapi_schema: Dict[str, Any]) -> str:
    """Render the OpenAPI schema as structured markdown."""
    paths = openapi_schema.get("paths", {})
    components = openapi_schema.get("components", {}).get("schemas", {})
    title = openapi_schema.get("info", {}).get("title", "API")
    version = openapi_schema.get("info", {}).get("version", "")

    # Group endpoints by tag
    by_tag: Dict[str, List[tuple[str, str, Dict[str, Any]]]] = {}
    for path, methods in paths.items():
        for method, spec in methods.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            tags = spec.get("tags", ["untagged"])
            for tag in tags:
                by_tag.setdefault(tag, []).append((path, method, spec))

    # Sort tags alphabetically, with "root" last
    tag_order = sorted(by_tag.keys(), key=lambda t: (t == "root", t))

    lines: List[str] = [
        f"# {title} API Reference",
        "",
        f"**Version:** {version}",
        "",
        f"**Base URL:** `http://localhost:8000`",
        "",
        f"**Total endpoints:** {sum(len(eps) for eps in by_tag.values())}",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
    ]
    for tag in tag_order:
        anchor = tag.lower().replace(" ", "-")
        lines.append(f"- [{tag}](#{anchor}) ({len(by_tag[tag])} endpoints)")
    lines.append("")
    lines.append("---")
    lines.append("")

    for tag in tag_order:
        lines.append(f"## {tag}")
        lines.append("")
        for path, method, spec in sorted(by_tag[tag], key=lambda x: (x[0], x[1])):
            summary = spec.get("summary", "")
            description = spec.get("description", "")
            operation_id = spec.get("operationId", "")
            lines.append(f"### {method.upper()} `{path}`")
            lines.append("")
            if summary:
                lines.append(f"**{summary}**")
                lines.append("")
            if description:
                lines.append(description)
                lines.append("")
            parameters = spec.get("parameters", [])
            if parameters:
                lines.append("#### Parameters")
                lines.append("")
                lines.append(_format_parameters(parameters, components))
                lines.append("")
            body = spec.get("requestBody")
            if body:
                lines.append("#### Request Body")
                lines.append("")
                lines.append(_format_request_body(body, components))
                lines.append("")
            responses = spec.get("responses", {})
            if responses:
                lines.append("#### Responses")
                lines.append("")
                lines.append(_format_responses(responses, components))
                lines.append("")
            lines.append("---")
            lines.append("")
        lines.append("")

    # Append schema reference section
    if components:
        lines.append("## Schema Reference")
        lines.append("")
        for schema_name in sorted(components.keys()):
            schema = components[schema_name]
            lines.append(f"### `{schema_name}`")
            lines.append("")
            props = schema.get("properties", {})
            required_fields = schema.get("required", [])
            if props:
                lines.append("| Field | Type | Required | Description |")
                lines.append("|-------|------|----------|-------------|")
                for prop_name, prop_schema in props.items():
                    req = "Yes" if prop_name in required_fields else "No"
                    ptype = prop_schema.get("type", "any")
                    if "$ref" in prop_schema:
                        ptype = _schema_name(prop_schema)
                    elif prop_schema.get("enum"):
                        ptype = f"{ptype} ({', '.join(prop_schema['enum'])})"
                    desc = prop_schema.get("description", "").replace("\n", " ") or ""
                    lines.append(f"| `{prop_name}` | {ptype} | {req} | {desc} |")
            else:
                lines.append("*No properties defined*")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate API docs from FastAPI OpenAPI schema")
    parser.add_argument("--output", "-o", default="docs/api.md", help="Output file path")
    args = parser.parse_args()

    from fastapi.openapi.utils import get_openapi

    # Import the app (this runs all the route definitions and tagging)
    import main as server_main

    openapi_schema = get_openapi(
        title=server_main.app.title,
        version=server_main.app.version,
        description=server_main.app.description,
        routes=server_main.app.routes,
    )

    markdown = generate_docs(openapi_schema)
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Generated {output_path} ({len(markdown)} bytes, {markdown.count(chr(10))} lines)")

    # Also dump the raw OpenAPI JSON for reference
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(openapi_schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {json_path}")


if __name__ == "__main__":
    main()
