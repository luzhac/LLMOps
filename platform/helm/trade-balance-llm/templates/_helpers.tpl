{{- define "trade-balance-llm.labels" -}}
app.kubernetes.io/part-of: trade-balance-llm
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end }}

