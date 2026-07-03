import json
import jsonschema
from jsonschema import validate
from pathlib import Path

MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "class_enum": {
            "type": "object",
            "patternProperties": {
                "^.*$": {
                    "type": "integer",
                }
            },
            "additionalProperties": False
        },
    },
    "nb_classes": {
        "type": "integer"
    },
    "nb_features": {
        "type": "integer"
    },
    "class_col": {
        "type": "string"
    },
    "trimeans": {
        "type": "array",
        "items": {
            "type": "number"
        }
    },
    "features_cols": {
        "type": "array",
        "items": {
            "type": "string"
        }
    },
    "weights": {
        "type": "array",
        "items": {
            "type": "array",
            "items": {
                "type": "number"
            }
        }
    },
    "biases": {
        "type": "array",
        "items": {
            "type": "array",
            "items": {
                "type": "number"
            }
        }
    },
    "required": [
        "class_enum",
        "nb_classes",
        "nb_features",
        "class_col",
        "trimeans",
        "features_cols",
        "weights",
        "biases"
    ],
    "$schema": "https://json-schema.org/draft/2020-12/schema"
}

def validate_json(file: Path) -> None:
    try:
        with open(file) as f:
            my_json = json.load(f)

        validate(instance=my_json, schema=MODEL_SCHEMA)

    except jsonschema.exceptions.ValidationError as ve:
        print(f"Validation error: {ve.message}")

    except jsonschema.exceptions.SchemaError as se:
        print(f"Schema error: {se.message}")

    except Exception as e:
        print(f"Error: {e}")