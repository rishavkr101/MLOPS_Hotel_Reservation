pipeline{
    agent any

    stages{
        stage('cloning Github repo to jenkins'){
            steps{
                script{
                    echo 'cloning Github repo to jenkins....'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/rishavkr101/MLOPS_Hotel_Reservation.git']])
                }
            }
        }

    }
}