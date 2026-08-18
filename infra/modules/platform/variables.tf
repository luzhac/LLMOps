variable "project_id" {
  description = "Immutable GCP project ID, not the display name."
  type        = string
}

variable "region" {
  description = "Region containing the zonal cluster."
  type        = string
}

variable "zone" {
  description = "Single zone for the cluster. Confirm accelerator availability first if enable_gpu_pool is true."
  type        = string
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
  description = "Must be unique within the project across every environment (dev, production, ...)."
  type        = string
}

variable "node_sa_account_id" {
  description = "GCP service-account account_id for GKE nodes. Must be unique per environment and <=30 characters."
  type        = string
}

variable "system_machine_type" {
  description = "CPU pool for gateway, Argo CD and lightweight monitoring."
  type        = string
  default     = "e2-standard-2"
}

variable "enable_gpu_pool" {
  description = "Set false to skip creating the GPU node pool entirely — no GPU quota is consumed, no stockout risk. Useful for a cheap dev environment that only exercises the gateway/system layer."
  type        = bool
  default     = true
}

variable "gpu_accelerator_type" {
  description = "GPU type, e.g. nvidia-tesla-t4 or nvidia-l4. Must be available in the chosen zone."
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_machine_type" {
  description = "Machine shape sized to carry one GPU of the chosen accelerator type."
  type        = string
  default     = "n1-standard-8"
}

variable "gpu_spot" {
  description = "Use interruptible Spot GPU capacity."
  type        = bool
  default     = false
}

variable "gpu_node_locations" {
  description = <<-EOT
    Zones the GPU node pool may place its single node in. Listing every zone in
    the region that offers the accelerator lets the cluster autoscaler fall back
    when one zone returns ZONE_RESOURCE_POOL_EXHAUSTED (a stockout). Capacity is
    still capped at one node in total by autoscaling.total_max_node_count, so
    adding zones increases availability without increasing cost.
  EOT
  type        = list(string)
  default     = []
}

variable "subnet_cidr" {
  type    = string
  default = "10.10.0.0/20"
}

variable "pods_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "services_cidr" {
  type    = string
  default = "10.30.0.0/20"
}

variable "master_ipv4_cidr_block" {
  description = "Private /28 for the GKE control plane. Must not overlap with another environment's if they ever share a network."
  type        = string
  default     = "172.16.0.0/28"
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
    managed-by  = "terraform"
  }
}
