terraform {
  required_version = ">= 1.7.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }

  # Bootstrap has no remote backend by design: it creates the state bucket that
  # every other environment then uses. Its own state stays local, on the
  # machine that runs it, and is applied by a human, never by CI.
}

provider "google" {
  project = var.project_id
  region  = var.region
}
