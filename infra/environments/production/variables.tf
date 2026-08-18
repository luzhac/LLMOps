variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "asia-southeast1"
}

variable "zone" {
  type    = string
  default = "asia-southeast1-a"
}

variable "admin_cidr" {
  type = string
}

variable "artifact_bucket_location" {
  type    = string
  default = "ASIA"
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
