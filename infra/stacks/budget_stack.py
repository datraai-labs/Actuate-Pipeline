"""BudgetStack — a monthly cost ceiling with email alerts.

Not in the AWS Architecture doc. Added deliberately, because §7's cost pragmatics are only
advice unless something actually watches the bill. Credits run out quietly; an alarm is the
difference between finding out by email and finding out from an invoice.

Deploy this FIRST, before the buckets and the database. It costs nothing and it is the only
resource here that can tell you the others have gone wrong.

Alerts fire at 80% of the ceiling (actual spend), and — more usefully — at 100% of the
*forecast*, which warns you mid-month that the trajectory is bad rather than after the fact.
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import aws_budgets as budgets
from constructs import Construct


class BudgetStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        monthly_limit_usd: float,
        alert_email: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        subscribers = [
            budgets.CfnBudget.SubscriberProperty(
                address=alert_email, subscription_type="EMAIL"
            )
        ]

        budgets.CfnBudget(
            self,
            "MonthlyCost",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name=f"actuate-{env_name}-monthly",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(
                    amount=monthly_limit_usd, unit="USD"
                ),
                # Scoped to resources tagged project=actuate, so this watches Actuate's
                # spend specifically rather than everything else in the account.
                cost_filters={"TagKeyValue": ["user:project$actuate"]},
            ),
            notifications_with_subscribers=[
                # 80% of ACTUAL spend — "you are most of the way through the budget".
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        comparison_operator="GREATER_THAN",
                        notification_type="ACTUAL",
                        threshold=80,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=subscribers,
                ),
                # 100% of FORECAST — the one that actually saves you. It fires mid-month
                # when the trajectory is bad, not after the money is already spent.
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        comparison_operator="GREATER_THAN",
                        notification_type="FORECASTED",
                        threshold=100,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=subscribers,
                ),
            ],
        )

        cdk.CfnOutput(self, "MonthlyLimitUsd", value=str(monthly_limit_usd))
        cdk.CfnOutput(self, "AlertEmail", value=alert_email)
