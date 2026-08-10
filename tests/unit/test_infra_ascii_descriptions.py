"""AWS rejects non-ASCII in description fields. Catch it at synth, not mid-deploy.

A real failure: an em-dash in the `PipelineRole` description failed `cdk deploy` after the
stack had already begun creating resources —

    Value at 'description' failed to satisfy constraint: Member must satisfy regular
    expression pattern: [\\u0009\\u000A\\u000D\\u0020-\\u007E\\u00A1-\\u00FF]

CloudFormation rolled back cleanly, but the round-trip cost a full deploy cycle, and the
DataStack's SecurityGroup `GroupDescription` had the identical bug waiting behind it.

`cdk synth` does not validate this; only the AWS API does. So this test does, offline.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parents[2] / "infra"

pytest.importorskip("aws_cdk", reason="aws-cdk-lib not installed")

#: The character class AWS actually permits in these fields.
_ALLOWED = re.compile(r"^[\x09\x0a\x0d\x20-\x7e\xa1-\xff]*$")

#: Properties AWS validates against that class. Not exhaustive across all of AWS, but
#: these are the ones our stacks set.
_DESCRIPTION_PROPS = ("Description", "GroupDescription")


@pytest.fixture(scope="module")
def templates() -> dict[str, dict]:
    sys.path.insert(0, str(INFRA))
    from aws_cdk import App
    from aws_cdk.assertions import Template
    from stacks.budget_stack import BudgetStack
    from stacks.data_stack import DataStack
    from stacks.storage_stack import StorageStack

    app = App(context={"env": "dev"})
    storage = StorageStack(app, "S", env_name="dev")
    data = DataStack(app, "D", env_name="dev", key=storage.key)
    budget = BudgetStack(
        app, "B", env_name="dev", monthly_limit_usd=50, alert_email="a@b.com"
    )
    return {
        "StorageStack": Template.from_stack(storage).to_json(),
        "DataStack": Template.from_stack(data).to_json(),
        "BudgetStack": Template.from_stack(budget).to_json(),
    }


def test_no_description_field_contains_characters_aws_will_reject(templates):
    offenders = []
    for stack, template in templates.items():
        for logical_id, resource in template.get("Resources", {}).items():
            props = resource.get("Properties", {})
            for prop in _DESCRIPTION_PROPS:
                value = props.get(prop)
                if isinstance(value, str) and not _ALLOWED.match(value):
                    bad = sorted({c for c in value if not _ALLOWED.match(c)})
                    offenders.append(
                        f"{stack}/{logical_id} ({resource['Type']}).{prop} "
                        f"contains {bad!r}: {value[:60]!r}"
                    )

    assert not offenders, (
        "AWS will reject these at deploy time (cdk synth does NOT catch it):\n  "
        + "\n  ".join(offenders)
        + "\n\nUse ASCII in any field AWS validates as a description."
    )
