data "google_billing_account" "current" {
  count = var.billing_account_id == null ? 0 : 1

  billing_account = var.billing_account_id
  lookup_projects = false

  depends_on = [google_project_service.required]
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

  depends_on = [google_project_service.required]
}
