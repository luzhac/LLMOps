resource "google_container_cluster" "main" {
  project  = var.project_id
  name     = var.cluster_name
  location = var.zone

  network    = google_compute_network.main.id
  subnetwork = google_compute_subnetwork.gke.id

  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false
  enable_shielded_nodes    = true

  release_channel {
    channel = "REGULAR"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = var.admin_cidr
      display_name = "administrator"
    }
  }

  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]

    managed_prometheus {
      enabled = false
    }
  }

  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }

    http_load_balancing {
      disabled = false
    }

    gcp_filestore_csi_driver_config {
      enabled = false
    }
  }

  maintenance_policy {
    recurring_window {
      start_time = "2026-01-04T02:00:00Z"
      end_time   = "2026-01-04T06:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SU"
    }
  }

  resource_labels = var.labels

  depends_on = [
    google_compute_router_nat.main,
    google_project_service.required,
  ]
}

resource "google_container_node_pool" "system" {
  project  = var.project_id
  name     = "system"
  location = var.zone
  cluster  = google_container_cluster.main.name

  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = var.system_machine_type
    disk_type       = "pd-balanced"
    disk_size_gb    = 50
    image_type      = "COS_CONTAINERD"
    service_account = google_service_account.gke_nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels          = merge(var.labels, { workload = "system" })

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

resource "google_container_node_pool" "gpu" {
  project  = var.project_id
  name     = "l4-spot"
  location = var.zone
  cluster  = google_container_cluster.main.name

  initial_node_count = 0

  # Spread the pool across every zone in the region that offers the L4 so a
  # single-zone stockout does not block scale-up. See var.gpu_node_locations.
  #
  # Known wrinkle: GKE rejects a single update that changes node_locations and
  # autoscaling together ("Failed to update node pool"). If apply fails on this
  # resource, run the location change once by hand and then re-apply:
  #   gcloud container node-pools update l4-spot \
  #     --cluster=trade-balance-llm --zone=europe-west4-a \
  #     --node-locations=europe-west4-a,europe-west4-b,europe-west4-c
  node_locations = var.gpu_node_locations

  # total_* caps the node count across ALL zones. With the per-zone
  # min_node_count/max_node_count form, max=1 would allow one node PER zone,
  # i.e. up to three L4s once node_locations lists three zones.
  # location_policy = ANY tells the autoscaler to take whichever zone currently
  # has capacity, which is exactly the stockout fallback behaviour we want.
  autoscaling {
    total_min_node_count = 0
    total_max_node_count = 1
    location_policy      = "ANY"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = var.gpu_machine_type
    disk_type       = "pd-balanced"
    disk_size_gb    = 100
    image_type      = "COS_CONTAINERD"
    spot            = var.gpu_spot
    service_account = google_service_account.gke_nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels          = merge(var.labels, { workload = "gpu-inference" })

    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1

      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  lifecycle {
    ignore_changes = [initial_node_count]
  }

  timeouts {
    create = "60m"
    update = "60m"
    delete = "60m"
  }
}

