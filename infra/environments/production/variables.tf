variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "europe-central2"
}

variable "zone" {
  type    = string
  default = "europe-central2-c"
}

variable "admin_cidr" {
  type = string
}

variable "artifact_bucket_location" {
  type    = string
  default = "EU"
}

variable "billing_account_id" {
  type     = string
  default  = null
  nullable = true
}

variable "budget_amount" {
  type    = number
  default = 30
}
