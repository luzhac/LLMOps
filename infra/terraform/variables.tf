variable "project_id" {
  description = "Immutable GCP project ID, not the display name."
  type        = string
}

variable "region" {
  description = "Region containing the zonal cluster. europe-central2 (Warsaw) has T4 capacity; it does not offer L4 at all."
  type        = string
  default     = "europe-central2"
}

variable "zone" {
  description = "Single zone for the cost-bounded development cluster. Confirm accelerator availability first."
  type        = string
  default     = "europe-central2-c"
}

variable "admin_cidr" {
  description = "Public IPv4 CIDR allowed to reach the GKE control plane, normally YOUR_IP/32."
  type        = string

  validation {
    condition     = can(cidrnetmask(var.admin_cidr)) && var.admin_cidr != "0.0.0.0/0"
    error_message = "Use a specific administrator CIDR such as 203.0.113.10/32; 0.0.0.0/0 is rejected."
  }
}

variable "cluster_name" {
  type    = string
  default = "trade-balance-llm"
}

variable "system_machine_type" {
  description = "CPU pool for gateway, Argo CD and lightweight monitoring."
  type        = string
  default     = "e2-standard-2"
}

variable "gpu_machine_type" {
  description = "N1 shape sized to carry one 16 GB NVIDIA T4."
  type        = string
  default     = "n1-standard-8"
}

variable "gpu_spot" {
  description = "Use interruptible Spot GPU capacity for the portfolio environment."
  type        = bool
  default     = true
}

variable "gpu_node_locations" {
  description = <<-EOT
    Zones the GPU node pool may place its single node in. Listing every zone in
    the region that offers the accelerator lets the cluster autoscaler fall back
    when one zone returns ZONE_RESOURCE_POOL_EXHAUSTED (a stockout). Capacity is
    still capped at one node in total by autoscaling.total_max_node_count, so
    adding zones increases availability without increasing cost.

    Verify the list before changing regions:
      gcloud compute accelerator-types list \
        --filter="name=nvidia-tesla-t4 AND zone~europe-central2" --format="table(zone)"
  EOT
  type        = list(string)
  default     = ["europe-central2-b", "europe-central2-c"]
}

variable "artifact_bucket_location" {
  type    = string
  default = "EU"
}

variable "billing_account_id" {
  description = "Billing account ID used only for budget creation. Leave null to skip the budget resource."
  type        = string
  default     = null
  nullable    = true
}

variable "budget_amount" {
  description = "Alerting threshold in the billing account's own currency; not a hard spending cap."
  type        = number
  default     = 30

  validation {
    condition     = var.budget_amount > 0
    error_message = "budget_amount must be greater than zero."
  }
}

variable "labels" {
  type = map(string)
  default = {
    application = "trade-balance-llm"
    environment = "portfolio"
    managed-by  = "terraform"
  }
}
