pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "infra-window-453718-m0"
        IMAGE_NAME = "gcr.io/${GCP_PROJECT}/mlops-project1"
        CLUSTER_NAME = "mlops-cluster"           // ← Update this
        ZONE = "us-central1-a"                   // ← Update this
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }

    stages {

        stage('Clone Repository') {
            steps {
                echo '📥 Cloning GitHub repository...'
                checkout scm
            }
        }

        stage('Setup Python Environment') {
            steps {
                echo '🐍 Setting up virtual environment...'
                sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                '''
            }
        }

        stage('Build & Push Docker Image to GCR') {
            steps {
                withCredentials([file(credentialsId: 'gcp_key', variable: 'GCP_KEY_FILE')]) {
                    echo '🐳 Building and Pushing Docker Image to GCR...'
                    sh '''
                        gcloud auth activate-service-account --key-file="${GCP_KEY_FILE}"
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet

                        docker build -t ${IMAGE_NAME}:latest .
                        docker push ${IMAGE_NAME}:latest
                    '''
                }
            }
        }

        stage('Deploy to GKE') {
            steps {
                withCredentials([file(credentialsId: 'gcp_key', variable: 'GCP_KEY_FILE')]) {
                    echo '🚀 Deploying to GKE...'
                    sh '''
                        gcloud auth activate-service-account --key-file="${GCP_KEY_FILE}"
                        gcloud config set project ${GCP_PROJECT}

                        gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE}

                        kubectl apply -f k8s/deployment.yaml
                        kubectl apply -f k8s/service.yaml

                        kubectl set image deployment/mlops-project1 \
                            mlops-project1=${IMAGE_NAME}:latest

                        kubectl rollout status deployment/mlops-project1
                    '''
                }
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check the logs above.'
        }
        always {
            sh 'rm -rf ${VENV_DIR}'
        }
    }
}