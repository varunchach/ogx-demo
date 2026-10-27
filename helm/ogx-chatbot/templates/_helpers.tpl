{{- define "ogx-chatbot.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ogx-chatbot.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- include "ogx-chatbot.name" . -}}
{{- end -}}
{{- end -}}

{{- define "ogx-chatbot.namespace" -}}
{{- .Values.namespace.name | default .Release.Namespace -}}
{{- end -}}

{{- define "ogx-chatbot.labels" -}}
app: {{ include "ogx-chatbot.fullname" . }}
app.kubernetes.io/name: {{ include "ogx-chatbot.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: ogx-chatbot
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "ogx-chatbot.selectorLabels" -}}
app: {{ include "ogx-chatbot.fullname" . }}
{{- end -}}

{{- define "ogx-chatbot.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- .Values.serviceAccount.name | default (include "ogx-chatbot.fullname" .) -}}
{{- else -}}
{{- .Values.serviceAccount.name | default "default" -}}
{{- end -}}
{{- end -}}
