# Billing budget lives in bootstrap, not in environments/production, on purpose.
#
# It is a set-once resource: the alert thresholds don't change between
# deployments. Keeping it here means the CI service account never needs any
# permission on the billing account, which is a separate IAM resource level
# from the project (a billing account pays for many projects, so project-level
# roles deliberately do not inherit into it). CI hitting a 403 on
# data.google_billing_account was the symptom that led to this move.
#
# Bootstrap runs locally under a human identity that already holds
# roles/billing.admin, so it can manage this without widening CI's access.
#
# NOTE: a budget only ALERTS. It does not cap or stop spending.

data "google_project" "current" {
  project_id = var.project_id
}

data "google_billing_account" "current" {
  count = var.billing_account_id == null ? 0 : 1

  billing_account = var.billing_account_id
  lookup_projects = false
}

resource "google_billing_budget" "project" {
  count = var.billing_account_id == null ? 0 : 1

  billing_account = data.google_billing_account.current[0].id
  display_name    = "${var.cluster_name} budget"

  amount {
    specified_amount {
      currency_code = data.google_billing_account.current[0].currency_code
      units         = tostring(var.budget_amount)
    }
  }

  budget_filter {
    projects = ["projects/${data.google_project.current.number}"]
  }

  threshold_rules {
    threshold_percent = 0.25
  }

  threshold_rules {
    threshold_percent = 0.50
  }

  threshold_rules {
    threshold_percent = 0.80
  }

  threshold_rules {
    threshold_percent = 1.00
  }
}
