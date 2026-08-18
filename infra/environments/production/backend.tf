# 2026-08-18: the old flat infra/terraform config was destroyed and this
# module-based environment is now authoritative, built fresh (not migrated).
terraform {
  backend "gcs" {
    bucket = "project-2759a06c-a804-420c-a8b-tfstate"
    prefix = "trade-balance-llm/production"
  }
}
