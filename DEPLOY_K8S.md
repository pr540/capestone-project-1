# Workflow to deploy to local Kubernetes
# 1. Build the docker image locally (Using BuildKit for speed)
$env:DOCKER_BUILDKIT=1
docker build -t speech-emotion-app:latest .

# 2. Apply the Kubernetes configuration
kubectl apply -f k8s/deployment.yaml

# 3. Wait for the pod to be ready
# kubectl get pods -w

# 4. Access the application
# On Docker Desktop for Windows, it should be available at http://localhost:5000

# TIP: For even faster rebuilds, Docker Desktop's "Enable BuildKit" should be checked in settings.
