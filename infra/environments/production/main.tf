module "platform" {
  source = "../../modules/platform"

  project_id = var.project_id
  region     = var.region
  zone       = var.zone
  admin_cidr = var.admin_cidr

  cluster_name = "trade-balance-llm"
  # Matches the account_id already in use by the real, currently-running
  # cluster — do not change this without a state migration plan.
  node_sa_account_id = "trade-balance-gke-nodes"

  enable_gpu_pool      = true
  gpu_accelerator_type = "nvidia-tesla-t4"
  gpu_machine_type     = "n1-standard-8"
  gpu_spot             = false
  gpu_node_locations   = ["europe-central2-b", "europe-central2-c"]

  artifact_bucket_location = var.artifact_bucket_location
  billing_account_id       = var.billing_account_id
  budget_amount            = var.budget_amount

  labels = {
    application = "trade-balance-llm"
    environment = "production"
    managed-by  = "terraform"
  }
}
