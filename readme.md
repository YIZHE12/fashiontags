# FashionTag

This is a consulting project to develop a web application to automatically creating fashion tags for images uploaded. The training data is 1 million labelled fashion images from a CVPR 2018 workshop. I used transfer learning to create embedding data of the fashion images to achieve more than 100 times accerlation in the deep neural network training process. Flask was used to develop the web application. The final product is dockerized and can be pull from docker hub. 


## Getting Started

To get the data, please go to the 'Data' folder in this repo for further instructions. To train your own model, you will first need to generate embedding data for the images and store the embedding in your data folder. This will acceccrate your deep neural network training process by trading time complexcity with space complexicity. 

All scripts should be ran from the project root directory, e.g.:

```
python scripts/training.py
```
### Prerequisites

I used tensorflow and keras for the model development on AWS enviroment tensorflow_p36. To install a similar enviroment, you can run pip to install the following packages. For the Flask development, you can use the docker file to setup the enviroment.

```
pip install tensorflow==1.13.1
pip install Keras==2.2.4
pip install Keras-Applications==1.0.7
pip install keras-metrics==1.1.0
pip install Keras-Preprocessing==1.0.9
```

## Running the tests

The code should be execute in the correct sequence. 

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

