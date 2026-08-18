output "cluster_name" {
  value = google_container_cluster.main.name
}

output "cluster_zone" {
  value = var.zone
}

output "get_credentials_command" {
  value = "gcloud container clusters get-credentials ${google_container_cluster.main.name} --zone ${var.zone} --project ${var.project_id}"
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.containers.repository_id}"
}

output "benchmark_bucket" {
  value = google_storage_bucket.artifacts.name
}

output "cost_warning" {
  value = "GPU spend continues while the vLLM Pod keeps the L4 node alive. Run make model-down after every session. Budgets alert; they do not cap spend."
}

