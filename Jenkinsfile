pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages{
        stage('cloning Github repo to jenkins'){
            steps{
                script{
                    echo 'cloning Github repo to jenkins....'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/rishavkr101/MLOPS_Hotel_Reservation.git']])
                }
            }
        }

        stage('setting up our virtual environment and installing dependancies'){
            steps{
                script{
                    echo 'setting up our virtual environment and installing dependancies......'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgarde pip
                    pip install -e .
                    '''

                }
            }
        }

    }
}