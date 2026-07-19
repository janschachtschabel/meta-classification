{{/*
Chart name (optionally overridden). Underscores are sanitized to hyphens so
the value is a valid Kubernetes (RFC 1123 / DNS-1035) resource-name component.
*/}}
{{- define "classification-api.name" -}}
{{- default .Chart.Name .Values.nameOverride | replace "_" "-" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name used for resource names. Following the edu-sharing
convention this is the chart name (or nameOverride/fullnameOverride) WITHOUT a
release-name prefix. Underscores are sanitized to hyphens for RFC 1123 / DNS-1035.
*/}}
{{- define "classification-api.fullname" -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- default $name .Values.fullnameOverride | replace "_" "-" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Chart label value. Deliberately WITHOUT the chart version: this label ends up on
the StatefulSet's immutable spec.volumeClaimTemplates, and the CI varies the chart
version per branch/tag (0.0.0-<slug>) — a changing value there breaks StatefulSet
updates ("forbidden: updates to statefulset spec ... are forbidden").
*/}}
{{- define "classification-api.chart" -}}
{{- .Chart.Name | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common metadata labels.
*/}}
{{- define "classification-api.labels" -}}
helm.sh/chart: {{ include "classification-api.chart" . }}
{{ include "classification-api.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels (stable across upgrades).
*/}}
{{- define "classification-api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "classification-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Fully qualified container image reference.
*/}}
{{- define "classification-api.image" -}}
{{- $repo := .Values.image.name -}}
{{- if not $repo -}}
{{- $repo = printf "%s/%s" .Values.global.image.registry .Values.global.image.repository -}}
{{- end -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
