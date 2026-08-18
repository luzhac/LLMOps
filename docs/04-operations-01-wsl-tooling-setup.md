# WSL deployment tool installation / WSL 部署工具安装

本文说明如何在 Windows 11 + WSL 2 + Ubuntu/Debian 中安装本项目需要的本地工具：

- Google Cloud CLI（`gcloud`）和 GKE authentication plugin；
- Terraform；
- kubectl；
- Helm 3；
- Docker Desktop 的 WSL 2 integration；
- Python、OpenSSL、curl、Git 和 Make。

命令默认在 **WSL Ubuntu Bash** 中执行。标记为 PowerShell 的命令必须在 Windows PowerShell 中执行。

> 本文安装的都是本地客户端。安装工具本身不会创建 GCP 资源，也不会产生 GCP GPU 费用。真正产生云费用的是后续的 Terraform apply 和启动 GPU Pod。

## 1. 确认 WSL 2 / Confirm WSL 2

先在 **Windows PowerShell** 中执行：

```powershell
wsl --version
wsl --list --verbose
```

Ubuntu 对应的 VERSION 必须是 `2`。如果不是：

```powershell
wsl --update
wsl --set-version Ubuntu 2
wsl --set-default-version 2
```

如果发行版名称不是 `Ubuntu`，使用 `wsl --list --verbose` 显示的真实名称。

进入 WSL：

```powershell
wsl
```

在 **WSL Bash** 中确认系统和 CPU 架构：

```bash
cat /etc/os-release
uname -m
dpkg --print-architecture
```

下文以 Ubuntu/Debian 和 `amd64` 为默认。ARM Windows/WSL 显示 `arm64` 时，kubectl 和 Helm 下载地址必须改成 arm64。

WSL 官方安装与更新说明：
https://learn.microsoft.com/windows/wsl/install

## 2. 安装通用依赖 / Base packages

在 WSL Bash 中：

```bash
sudo apt-get update
sudo apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  software-properties-common \
  wget \
  unzip \
  git \
  make \
  python3 \
  python3-venv \
  python3-pip \
  openssl
```

验证：

```bash
git --version
make --version
python3 --version
openssl version
curl --version
```

## 3. 安装 Google Cloud CLI / Install gcloud

添加 Google 官方 apt key 和软件源：

```bash
sudo mkdir -p -m 0755 /usr/share/keyrings

curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

sudo apt-get update
sudo apt-get install -y \
  google-cloud-cli \
  google-cloud-cli-gke-gcloud-auth-plugin
```

验证：

```bash
gcloud version
gke-gcloud-auth-plugin --version
```

然后登录。浏览器通常会在 Windows 中打开：

```bash
gcloud auth login
gcloud auth application-default login
```

两条命令不能互相替代：

- `gcloud auth login`：供 gcloud CLI 使用；
- `gcloud auth application-default login`：供本地 Terraform Google provider 使用。

设置项目后，再设置 ADC quota project：

```bash
export GCP_PROJECT_ID='replace-with-your-project-id'
gcloud config set project "$GCP_PROJECT_ID"
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
gcloud auth list
gcloud config list
```

不要为本地个人部署下载 service-account JSON key。

官方安装说明：
https://cloud.google.com/sdk/docs/install-sdk

Terraform 本地认证说明：
https://cloud.google.com/docs/terraform/authentication

## 4. 安装 Terraform / Install Terraform

添加 HashiCorp 官方签名和 apt 软件源：

```bash
wget -O- https://apt.releases.hashicorp.com/gpg \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg >/dev/null

gpg --no-default-keyring \
  --keyring /usr/share/keyrings/hashicorp-archive-keyring.gpg \
  --fingerprint

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(grep -oP '(?<=UBUNTU_CODENAME=).*' /etc/os-release || lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/hashicorp.list

sudo apt-get update
sudo apt-get install -y terraform
```

验证版本和本项目 Terraform：

```bash
terraform version
terraform -chdir=infra/terraform fmt -check -recursive
terraform -chdir=infra/terraform init -backend=false
terraform -chdir=infra/terraform validate
```

`terraform init` 会从 Terraform Registry 下载 provider，因此需要网络。它不会创建 GCP 资源；`terraform apply` 才会创建资源。

官方安装说明：
https://developer.hashicorp.com/terraform/install

## 5. 安装 kubectl / Install kubectl

本项目使用 GKE。kubectl 客户端最好与集群版本相差不超过一个 minor version。下面的软件源固定为 Kubernetes v1.36，这是编写本文时的稳定 minor；如果官方稳定版本或 GKE 版本变化，把 URL 中的 `v1.36` 换成与集群兼容的 minor。

```bash
sudo mkdir -p -m 0755 /etc/apt/keyrings

curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.36/deb/Release.key \
  | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

sudo chmod 0644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.36/deb/ /' \
  | sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo chmod 0644 /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl
```

验证：

```bash
kubectl version --client --output=yaml
```

创建 GKE 后可查看 server/client 版本：

```bash
kubectl version
```

如果版本差距超过一个 minor，按 Kubernetes 官方页面选择与 GKE server 匹配的软件源。

官方安装和版本兼容说明：
https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/

## 6. 安装 Helm 3 / Install Helm 3

项目当前按 Helm 3 编写。这里安装并校验固定的 Helm 3 release，避免 apt 后续自动切换 major version。

以下命令假设 `amd64`：

```bash
export HELM_VERSION='v3.21.1'
export HELM_ARCH='amd64'
export HELM_ARCHIVE="helm-$HELM_VERSION-linux-$HELM_ARCH.tar.gz"
export HELM_TMP_DIR="$(mktemp -d)"

curl -fsSLo "$HELM_TMP_DIR/$HELM_ARCHIVE" \
  "https://get.helm.sh/$HELM_ARCHIVE"

curl -fsSLo "$HELM_TMP_DIR/$HELM_ARCHIVE.sha256sum" \
  "https://get.helm.sh/$HELM_ARCHIVE.sha256sum"

(
  cd "$HELM_TMP_DIR"
  sha256sum --check "$HELM_ARCHIVE.sha256sum"
  tar -xzf "$HELM_ARCHIVE"
)

sudo install -o root -g root -m 0755 \
  "$HELM_TMP_DIR/linux-$HELM_ARCH/helm" /usr/local/bin/helm
```

ARM64 把 `HELM_ARCH` 改成 `arm64`。验证：

```bash
helm version
helm lint platform/helm/trade-balance-llm
helm template trade-balance-llm platform/helm/trade-balance-llm \
  --namespace llm-platform >/dev/null
```

官方安装说明：
https://helm.sh/docs/v3/intro/install/

Helm 3/Kubernetes 版本兼容表：
https://helm.sh/docs/v3/topics/version_skew/

## 7. 安装 Docker Desktop 并启用 WSL / Install Docker

推荐使用 **Docker Desktop for Windows + WSL 2 backend**。不要同时在同一个 WSL 发行版中安装另一套 Docker Engine；两套 daemon/socket 容易冲突。

### 7.1 Windows 中安装

1. 从官方页面下载 Docker Desktop for Windows：
   https://docs.docker.com/desktop/setup/install/windows-install/
2. 运行安装程序，选择 WSL 2 backend。
3. 启动 Docker Desktop。
4. 打开 **Settings > General**，确认使用 WSL 2 based engine。
5. 打开 **Settings > Resources > WSL Integration**。
6. 为当前 Ubuntu 发行版启用 integration，然后选择 **Apply & restart**。
7. 确认 Docker Desktop 当前运行 Linux containers。

Docker 官方要求 WSL 至少为 2.1.5，并建议使用最新版本：
https://docs.docker.com/desktop/features/wsl/

### 7.2 WSL 中验证

重新打开 WSL Bash：

```bash
docker version
docker info
docker run --rm hello-world
```

`docker version` 必须同时显示 Client 和 Server。如果只有 Client 或出现：

```text
Cannot connect to the Docker daemon
```

检查：

1. Windows 中 Docker Desktop 是否已启动；
2. 当前 WSL 发行版是否启用了 WSL Integration；
3. 是否误装了第二套 Linux Docker daemon；
4. 执行 `wsl --shutdown` 后重新打开 WSL 和 Docker Desktop。

不需要在 WSL 中运行 `sudo service docker start`；Docker Desktop 模式下 daemon 由 Windows Docker Desktop 管理。

## 8. 一次性验证所有工具 / Verify everything

在仓库根目录执行：

```bash
gcloud version
gke-gcloud-auth-plugin --version
terraform version
kubectl version --client
helm version
docker version
python3 --version
openssl version

bash -n scripts/preflight.sh scripts/build-gateway.sh scripts/model-session.sh
make test
make terraform-check
make helm-check
```

如果 `make test` 提示缺少 Python 包，使用独立 virtual environment：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r app/gateway/requirements-dev.txt
pip install -r loadtest/requirements.txt
make test
```

不要使用 `sudo pip install`，也不要把项目依赖安装进系统 Python。

## 9. 常见 WSL 问题 / Troubleshooting

### gcloud 登录没有自动打开浏览器

复制终端显示的登录 URL 到 Windows 浏览器完成认证。不要把认证码、token 或 ADC 文件提交到 Git。

### gke-gcloud-auth-plugin 找不到

```bash
sudo apt-get update
sudo apt-get install -y google-cloud-cli-gke-gcloud-auth-plugin
gke-gcloud-auth-plugin --version
```

### Terraform 找不到 ADC

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
```

### kubectl 无法连接

安装工具不等于已经连接集群。Terraform 创建 GKE 后还要执行：

```bash
gcloud container clusters get-credentials trade-balance-llm \
  --zone europe-west4-a \
  --project "$GCP_PROJECT_ID"

kubectl config current-context
kubectl get nodes
```

### Docker build 很慢

当前仓库位于 `/mnt/c`，Docker 在跨 Windows/WSL 文件系统读取大量小文件时可能较慢。这个项目的 gateway 很小，通常仍可接受。如果未来模型或 build context 很大，可以把工作副本放在 WSL Linux 文件系统中的专用项目目录。

### apt key 已存在导致命令失败

不要盲目覆盖来源不明的 key。先确认对应 source list 和 keyring 是否就是本文创建的官方源；如果工具已经可以正常验证版本，可以跳过重复添加软件源的步骤。

## 10. 下一步 / Next step

所有工具验证通过后，回到 [GCP 端到端 runbook](04-operations-03-runbook.md)，从 Project、Billing、API、quota preflight 开始。不要直接跳到 `terraform apply`，也不要在 GPU quota 为零时反复创建集群。
