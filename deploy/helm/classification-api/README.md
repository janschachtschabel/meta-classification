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
helm install classify deploy/helm/classification-api \
  --set config.auth.adminKey=<strong-random-key> \
  --set config.auth.readonlyKey=<strong-random-key> \
  --set ingress.hosts[0]=classify.example.de
```

`config.auth.adminKey` and `config.auth.readonlyKey` are **required** while
`config.auth.enabled=true` — rendering fails without them. They are stored in the
chart-managed `Secret` and map to the app's `X-API-Key` roles (admin = train/manage,
readonly = predict/status). Swagger UI: `https://<host>/docs`.

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
| `config.auth.enabled`                   | Enable API-key authentication                                            | `true`        |
| `config.auth.adminKey`                  | Admin API key (**REQUIRED** when auth enabled, stored in Secret)          | `""`          |
| `config.auth.readonlyKey`               | Readonly API key (**REQUIRED** when auth enabled, stored in Secret)       | `""`          |
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
| `resources.requests.memory`                 | Memory request                                                     | `1Gi`                |

For predict-only or small-data deployments, `resources.limits` of `1000m` / `2Gi`
are sufficient — training is what needs the headroom.
