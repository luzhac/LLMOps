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

  # billingbudgets.googleapis.com rejects plain user Application Default
  # Credentials with "requires a quota project, which is not set by default".
  # These two settings bill the API call to this project and are what make the
  # budget resource work — environments/production already had them.
  billing_project       = var.project_id
  user_project_override = true
}
