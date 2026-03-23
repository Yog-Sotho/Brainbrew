"""
Alpaca format exporter module.
Handles conversion of distillation output to Alpaca dataset format.
"""
import json
import os
import structlog
from pathlib import Path
from typing import Optional

logger = structlog.get_logger(__name__)


class ExporterError(Exception):
    """Custom exception for export failures."""
    pass


def validate_input_path(input_path: str) -> Path:
    """
    Validate the input file path exists and is readable.

    Args:
        input_path: Path to the input JSONL file

    Returns:
        Validated Path object

    Raises:
        ExporterError: If validation fails
    """
    path = Path(input_path)

    if not path.exists():
        raise ExporterError(f"Input file does not exist: {input_path}")

    if not path.is_file():
        raise ExporterError(f"Input path is not a file: {input_path}")

    if path.stat().st_size == 0:
        raise ExporterError(f"Input file is empty: {input_path}")

    return path


def validate_output_path(output_path: str) -> Path:
    """
    Validate the output path is writable.

    Args:
        output_path: Path for the output file

    Returns:
        Validated Path object

    Raises:
        ExporterError: If validation fails
    """
    path = Path(output_path)

    # Check parent directory exists and is writable
    parent = path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise ExporterError(f"Cannot create output directory: {e}")
        except Exception as e:
            raise ExporterError(f"Failed to create output directory: {e}")

    # Check if we can write to the location
    try:
        # Try to create the file (will be overwritten)
        path.touch()
    except PermissionError:
        raise ExporterError(f"Cannot write to output path: {output_path}")
    except Exception as e:
        raise ExporterError(f"Invalid output path: {e}")

    return path


def validate_record(obj: dict, line_number: int) -> dict:
    """
    Validate and extract required fields from a JSON record.

    Args:
        obj: Parsed JSON object
        line_number: Line number for error reporting

    Returns:
        Validated record with required fields

    Raises:
        ExporterError: If validation fails
    """
    if not isinstance(obj, dict):
        raise ExporterError(f"Line {line_number}: Expected JSON object, got {type(obj).__name__}")

    if "instruction" not in obj:
        raise ExporterError(f"Line {line_number}: Missing required field 'instruction'")

    instruction = obj.get("instruction")
    if not instruction or (isinstance(instruction, str) and not instruction.strip()):
        raise ExporterError(f"Line {line_number}: Field 'instruction' is empty")

    return {
        "instruction": str(instruction),
        "input": "",
        "output": str(obj.get("output", ""))
    }


def export_alpaca(input_path: str, output_path: str) -> int:
    """
    Export distillation output to Alpaca format.

    Converts JSONL with 'instruction' and 'output' fields to Alpaca format:
    {"instruction": ..., "input": "", "output": ...}

    Args:
        input_path: Path to input JSONL file
        output_path: Path for output JSONL file

    Returns:
        Number of records exported

    Raises:
        ExporterError: If export fails
    """
    logger.info("Starting export", input=input_path, output=output_path)

    # Validate paths
    input_path = validate_input_path(input_path)
    output_path = validate_output_path(output_path)

    records_exported = 0
    errors = []

    try:
        with open(input_path, encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line_num, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    obj = json.loads(line)
                    record = validate_record(obj, line_num)
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_exported += 1

                except json.JSONDecodeError as e:
                    error_msg = f"Line {line_num}: Invalid JSON - {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)

                except ExporterError as e:
                    logger.warning(str(e))
                    errors.append(str(e))

    except PermissionError as e:
        raise ExporterError(f"Permission denied reading input file: {e}")
    except UnicodeDecodeError as e:
        raise ExporterError(f"Input file is not valid UTF-8: {e}")
    except IOError as e:
        raise ExporterError(f"IO error during export: {e}")

    if records_exported == 0:
        error_summary = "; ".join(errors[:5]) if errors else "No valid records found"
        raise ExporterError(f"Export failed: {error_summary}")

    logger.info(
        "Export completed",
        records_exported=records_exported,
        errors=len(errors),
        error_summary=errors[:3] if errors else None
    )

    return records_exported
