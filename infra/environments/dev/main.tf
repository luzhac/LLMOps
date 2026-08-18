module "platform" {
  source = "../../modules/platform"

  project_id = var.project_id
  region     = var.region
  zone       = var.zone
  admin_cidr = var.admin_cidr

  cluster_name       = "trade-balance-llm-dev"
  node_sa_account_id = "tb-llm-dev-gke-nodes"

  # Cost-safe default: dev never creates a GPU node pool at all, so it can't
  # hit GPU quota or a regional stockout. It's for exercising the gateway,
  # networking, and Kubernetes layer only. Flip to true only for a deliberate,
  # time-boxed GPU test in dev.
  enable_gpu_pool = false

  subnet_cidr            = "10.11.0.0/20"
  pods_cidr              = "10.21.0.0/16"
  services_cidr          = "10.31.0.0/20"
  master_ipv4_cidr_block = "172.16.1.0/28"

  artifact_bucket_location = var.artifact_bucket_location
  billing_account_id       = var.billing_account_id
  budget_amount            = var.budget_amount

  labels = {
    application = "trade-balance-llm"
    environment = "dev"
    managed-by  = "terraform"
  }
}
