# edu_sharing-projects-wlo-classification-api

Helm chart for **MetaClassify** — a single-container FastAPI service
(TF-IDF + LogisticRegression metadata text classification, CPU-only, torch-free,
served on port `8000`). State lives on one persistent `/data` volume: uploaded CSV
datasets, trained model bundles, share links, the corrections editors contribute
(`POST /feedback`) and the history of finished training runs.

> **Single replica only.** The training job, model LRU cache and rate limiter are
> process-local, and the model store lives on one PVC — the chart deploys a
> `StatefulSet` with `replicaCount: 1`. Do not scale this chart horizontally.

## Install

```bash
# Recommended: the keys live in a Secret this chart does not own.
kubectl create secret generic classify-api-keys \
  --from-literal=APIV3_API_KEY_ADMIN=<strong-random-key> \
  --from-literal=APIV3_API_KEY_READONLY=<strong-random-key>

# Neither the tls values nor forwardedAllowIps are optional — see the two notes below.
helm install classify deploy/helm/classification-api \
  --set config.auth.existingSecret=classify-api-keys \
  --set config.limits.forwardedAllowIps=10.42.0.0/16 \
  --set ingress.hosts[0]=classify.example.de \
  --set ingress.tls[0].hosts[0]=classify.example.de \
  --set ingress.tls[0].secretName=classify-api-tls
```

`config.auth.existingSecret` names a `Secret` carrying `APIV3_API_KEY_ADMIN` and
`APIV3_API_KEY_READONLY`. The chart then renders no `Secret` of its own, which is what
sealed-secrets, External Secrets Operator and Vault need, and rotating a key becomes a
pod restart instead of a chart upgrade.

Without it, `config.auth.adminKey` and `config.auth.readonlyKey` are **required** while
`config.auth.enabled=true` — rendering fails without them — and are stored in the
chart-managed `Secret`. That path is fine for a throwaway cluster and poor beyond one: a
key passed with `--set` ends up in shell history, in the log of whatever CI ran the
command, and in any values file used to install. Changing them is a restart on this path
too: the pod carries no hash of the keys (anyone allowed to read pods could test guesses
against it), so an upgrade that only changes keys rolls nothing — follow it with
`kubectl rollout restart statefulset/<release>`, or change a `podAnnotations` value in the
same upgrade. Either way the keys map to the app's
`X-API-Key` roles (admin = train/manage, readonly = predict/status). Swagger UI:
`https://<host>/docs`.

> **TLS is not optional here.** Every authenticated call sends `X-API-Key` as a plain
> header, so an ingress without TLS publishes the credential to anything on the network path.
> With `ingress.enabled: true` the chart therefore refuses to render until either
> `ingress.tls` is filled in or `ingress.allowInsecure: true` says TLS is terminated above
> the ingress (a service mesh, a cloud load balancer) — something the chart cannot detect.

> **Name the proxy the rate limiter may believe.** Behind the ingress every request reaches
> the pod from the controller, so unless uvicorn may take the client address from
> `X-Forwarded-For`, all clients share one rate-limit bucket and one busy client throttles
> everyone. With `ingress.enabled` and `config.limits.rateLimitEnabled` the chart therefore
> refuses to render without `config.limits.forwardedAllowIps` — the controller's addresses,
> as narrow as you can name them (`10.42.0.0/16` above is k3s's whole pod range: every pod in
> it may then claim any client address, so pair a range like that with a NetworkPolicy that
> admits only the controller). `"*"` is refused. If you cannot name the controller, set
> `config.limits.rateLimitEnabled=false` and limit at the ingress instead.

## Parameters

### Global parameters

| Name                                       | Description                                            | Value                             |
| ------------------------------------------ | ------------------------------------------------------ | --------------------------------- |
| `global.annotations`                       | Define global annotations added to every pod           | `{}`                              |
| `global.cluster.cert.annotations`          | Set custom global cert annotations                     | `{}`                              |
| `global.cluster.domain`                    | Set global domain for the cluster                      | `cluster.local`                   |
| `global.cluster.ingress.ingressClassName`  | Set global ingress class name                          | `nginx`                           |
| `global.cluster.pdb.enabled`               | Enable PodDisruptionBudget                             | `false`                           |
| `global.debug`                             | Enable global debugging                                | `false`                           |
| `global.image.pullPolicy`                  | Set global image pullPolicy                            | `IfNotPresent`                    |
| `global.image.pullSecrets`                 | Set global image pullSecrets                           | `[]`                              |
| `global.image.registry`                    | Set global image container registry                    | `docker.edu-sharing.com`          |
| `global.image.repository`                  | Set global image container repository                  | `projects/wlo/classification-api` |
| `global.metrics.servicemonitor.enabled`    |  Enable a Prometheus ServiceMonitor (public `GET /metrics`)| `false`                           |
| `global.security`                          | Custom pod security context (merged)                   | `{}`                              |

### Local parameters

| Name               | Description                                                        | Value                          |
| ------------------ | ------------------------------------------------------------------ | ------------------------------ |
| `nameOverride`     | Override the chart name used for resource names                    | `""`                           |
| `fullnameOverride` | Fully override the generated resource name                         | `""`                           |
| `image.name`       | Override image repository (defaults to registry/repository)        | `""`                           |
| `image.tag`        | Set image tag (defaults to `.Chart.AppVersion`)                    | `""`                           |
| `replicaCount`     | Amount of replicas — MUST stay 1 (process-local state, one PVC)    | `1`                            |
| `service.type`     | Set service type                                                   | `ClusterIP`                    |
| `service.port`     | Set service port (cluster-internal)                                | `8000`                         |
| `ingress.enabled`  | Enable ingress                                                     | `true`                         |
| `ingress.hosts`    | Set ingress hosts                                                  | `["classify.127.0.0.1.nip.io"]`|
| `ingress.paths`    | Set paths served from the host                                     | `["/"]`                        |
| `ingress.tls`      | Set TLS for ingress                                                | `[]`                           |
| `ingress.annotations.*` | nginx body-size (≥ `config.limits.maxUploadMb`) / proxy timeouts | see `values.yaml`           |
| `debug`            | Enable debugging for this release                                  | `false`                        |

### Application configuration (env prefix `APIV3_`)

| Name                                    | Description                                                              | Value         |
| --------------------------------------- | ------------------------------------------------------------------------ | ------------- |
| `ingress.allowInsecure`                 | Allow an ingress with no TLS (see below)                                  | `false`       |
| `config.auth.enabled`                   | Enable API-key authentication; `false` serves `kubectl port-forward` only (refused with an ingress) | `true`        |
| `config.auth.existingSecret`            | Secret holding both keys; set this instead of the two below               | `""`          |
| `config.auth.adminKey`                  | Admin API key (**REQUIRED** when auth enabled and no existingSecret)      | `""`          |
| `config.auth.readonlyKey`               | Readonly API key (**REQUIRED** when auth enabled and no existingSecret)   | `""`          |
| `config.app.corsOrigins`                | Comma-separated allowed browser origins (empty = none)                   | `""`          |
| `config.app.logLevel`                   | Log level                                                                | `INFO`        |
| `config.compute.nJobs`                  | CPU cores for training (`auto` = all the pod may use, `-2` leave one free) | `auto`      |
| `config.compute.trainMemoryMb`          | Training memory budget in MiB (`auto` = 85 % of the memory limit, `0` = none) | `auto`   |
| `config.compute.trainingIsolation`      | Where a training runs (`process` = child process, `thread` = API process) | `process`    |
| `config.compute.solver`                 | Linear-head solver (float32-preserving `newton-cg`)                      | `newton-cg`   |
| `config.compute.parallelBackend`        | joblib backend (threading = one shared matrix)                           | `threading`   |
| `config.compute.tfidfMaxWordFeatures`   | Word n-gram feature cap (RAM lever)                                      | `80000`       |
| `config.compute.tfidfMaxCharFeatures`   | Char n-gram feature cap (RAM lever)                                      | `120000`      |
| `config.limits.maxUploadMb`             | Upload cap in MB (keep ingress body-size in sync)                        | `200`         |
| `config.limits.maxModelsInMemory`       | Trained models kept resident (LRU)                                       | `2`           |
| `config.limits.rateLimitEnabled`        | Enable the in-process rate limiter                                       | `true`        |
| `config.limits.forwardedAllowIps`       | Addresses whose `X-Forwarded-For` uvicorn believes: the ingress controller's. **Required** with ingress and rate limiter on (see below) | `""` |
| `config.extraEnv`                       | Extra plain environment variables (map)                                  | `{}`          |

### Storage, scheduling & runtime

| Name                                        | Description                                                       | Value                |
| ------------------------------------------- | ------------------------------------------------------------------ | -------------------- |
| `persistence.enabled`                       | Enable persistent storage for `/data` (all writable paths)         | `true`               |
| `persistence.mountPath`                     | Mount path for the data volume                                     | `/data`              |
| `persistence.storageClassName`              | StorageClass for the data PVC (empty → cluster default)            | `""`                 |
| `persistence.accessModes`                   | Access modes for the data PVC                                      | `["ReadWriteOnce"]`  |
| `persistence.size`                          | Storage request (datasets + ~250 MB per model bundle)              | `5Gi`                |
| `nodeAffinity` / `tolerations`              | Scheduling controls                                                | `{}` / `[]`          |
| `podAnnotations`                            | Set custom pod annotations                                         | `{}`                 |
| `podSecurityContext.fsGroup`                | fs group for volume access (image's `appuser`, uid/gid 1000)       | `1000`               |
| `securityContext.runAsUser`                 | User id to run as (image's `appuser`)                              | `1000`               |
| `securityContext.*`                         | non-root, no privilege escalation, drop ALL capabilities           | see `values.yaml`    |
| `terminationGracePeriod`                    | Grace period in seconds (training cancels cooperatively)           | `60`                 |
| `startupProbe.*` / `livenessProbe.*` / `readinessProbe.*` | Probe tuning (`GET /health`)                        | see `values.yaml`    |
| `resources.limits.cpu`                      | CPU limit (bounds training parallelism)                            | `4000m`              |
| `resources.limits.memory`                   | Memory limit (sized for ~600k-row training)                        | `8Gi`                |
| `resources.requests.cpu`                    | CPU request                                                        | `500m`               |
| `resources.requests.memory`                 | Memory request — equal to the limit, which training plans with      | `8Gi`                |

For predict-only or small-data deployments, `resources.limits` of `1000m` / `2Gi`
are sufficient — training is what needs the headroom. Lower `resources.requests.memory`
with the limit: a training plans with 85 % of the limit, so a smaller request lets the
scheduler place the pod where that memory is not free, and node pressure evicts it first.
