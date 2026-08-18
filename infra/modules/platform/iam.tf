resource "google_service_account" "gke_nodes" {
  project = var.project_id
  # GCP account_id has a 30-character limit, so this is a separate variable
  # instead of being derived from cluster_name (which would risk overflowing
  # it once an environment suffix like "-dev" is added).
  account_id   = var.node_sa_account_id
  display_name = "Least-privilege GKE node service account for ${var.cluster_name}"
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
