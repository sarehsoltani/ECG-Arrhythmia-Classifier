# Deployment Instructions

### Package the App into Docker

The prediction script has been embedded into a Flask application and packaged into a Docker image. To build the Docker image, run the following command:

```bash
docker build -t heart:v1 .
```

###Run the Docker Container
Once the Docker image is built, you can run it locally with the following command:
```bash
docker run -it --rm -p 9696:9696 heart:v1
```
This command will start the container and expose it on port 9696. You can access your Flask app at http://localhost:9696.
