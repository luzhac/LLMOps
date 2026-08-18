resource "google_service_account" "gke_nodes" {
  project      = var.project_id
  account_id   = "trade-balance-gke-nodes"
  display_name = "Least-privilege GKE node service account"
}

locals {
  node_roles = toset([
    "roles/artifactregistry.reader",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
  ])
}

resource "google_project_iam_member" "gke_nodes" {
  for_each = local.node_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

