# MnistDockerProject
This project aims to deploy a Mnist Project on the Docker container and store the predict result into the Cassandra database.

## Docker Preparation
To create a proper network to link the two container, create a network bridge first with the following command:
```
docker network create --driver=bridge --subnet=192.168.2.0/24 mnist
```

## Cassandra
You can directly pull Cassandra Docker images from DockerHub with the following command:
```
docker run --name {UserName}-cassandra --network=mnist --ip 192.168.2.100 -p 9042:9042 -d cassandra
```
(Option)Link to the Cassandra daemon if you want
```
docker run -it --link {UserName}-cassandra:cassandra --net mnist --rm cassandra cqlsh cassandra
```
## Mnist Project
Run the following command to build the docker image:
```
docker build -t "mnist-project" .
```
And run the project:
```
docker run -p 4000:80 --network=mnist --ip 192.168.2.10 mnist-project
```
## Check the result image in the cassandra database
Link to the Cassandra
```
docker run -it --link {UserName}-cassandra:cassandra --net mnist --rm cassandra cqlsh cassandra
```
Check the image with the following commands:
```
cqlsh> use mnistimages;
cqlsh:mnistimages> select * from images;
```
