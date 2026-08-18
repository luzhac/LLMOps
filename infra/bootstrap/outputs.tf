output "workload_identity_provider" {
  description = "Value for the GCP_WORKLOAD_IDENTITY_PROVIDER GitHub Actions variable."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "service_account_email" {
  description = "Value for the GCP_SERVICE_ACCOUNT GitHub Actions variable."
  value       = google_service_account.github_actions.email
}

output "state_bucket_name" {
  description = "Use this in each environment's backend.tf as the GCS bucket name."
  value       = google_storage_bucket.terraform_state.name
}

output "next_steps" {
  value = <<-EOT
    1. In the GitHub repo (Settings > Secrets and variables > Actions), Variables tab, set:
         GCP_REGION = ${var.deployment_region}
    2. Same place, Secrets tab, set:
         GCP_WORKLOAD_IDENTITY_PROVIDER = ${google_iam_workload_identity_pool_provider.github.name}
         GCP_SERVICE_ACCOUNT            = ${google_service_account.github_actions.email}
         GCP_PROJECT_ID                 = ${var.project_id}
    3. In each environment's backend.tf, point the GCS backend at bucket
       "${google_storage_bucket.terraform_state.name}" with a unique prefix per environment.
    4. Create the "production" environment under Settings > Environments and
       add required reviewers before wiring up deploy.yml.
  EOT
}
