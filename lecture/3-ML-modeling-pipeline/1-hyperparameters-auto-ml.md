## Hyperparameters tuning - Keras tuner 

### NAS - Neural Architecture Search
* NAS is a technique for automating the design of artificial neural networks
* It helps finding the optimal architecture
* This is a search over a huge space 
* AutoML is an algorithm to automate this search

### Types of parameters in ML Models 
* Trainable parameters:
    * Learned by the algorithm during training 
    * e.g. weights of a neural network 
* Hyperparameters:
    * Set before launching the learning process 
    * not updated in each training step 
    * e.g.: learning rate or the number of units in a dense layer 

### Manual hyperparameter tuning is not scalable
* Hyperparameters can be numerous even for small models 
* e.g. shallow DNN:
    * Architecture choices 
    * activation functions 
    * Weight initialization strategy 
    * Optimization hyperparameters such as learning rate, stop condition
* Tuning them manually can be a real brain teaser 
* Tuning helps with model performance 

### Automating hyperparameters tuning with Keras Tuner 
* Automation is key: open source ressources to the rescue 
* Keras Tuner:
    * Hyperparameter tuning with tensorflow2.0
    * Many methods available


## Is the architecture optimal? 
* Do the model need more or less hidden units to perform well?
* How does model size affect the convergence speed? 
* Is there any trade of between convergence speed, model size and accuracy?
* Search automation is the natural path to take 
* Keras tuner built in search functionality


![keras-tuner-ml-model](assets/keras-tuner-ml-model.png)
![keras-tuner-search-strategy](assets/keras-tuner-search-strategy.png)
![keras-tuner-callback](assets/keras-tuner-callback.png)

# AutoML - Automated Machine Learning 

![auto-ml](assets/auto-ml.png)

## Neural Architecture Search
![neural-architecture-strategy](assets/neural-architecture-strategy.png)

* **AutoML** automates the development of ML models 
* **AutoML** is not specific to a particular type of model
* Neural Architecture Search (**NAS**) is a subfield of AutoML
* NAS is a technique for automating the desing of Artificial neural networks (ANN) ![NAS](assets/NAS.png)

### Real-world use: Meredith Digital
![real-world-example-meredith-digital](assets/real-world-example-meredith-digital.png)

### Understanding search spaces 
Two Types of search spaces:
![types-of-search-spaces](assets/types-of-search-spaces.png)

* #### Macro Architecture search space![macro-search-space](assets/macro-search-space.png)

* #### Micro Architecture Search Space ![micro-search-space](assets/micro-search-space.png)

### Search Strategies 

**A few Search Strategies:**
1. **Grid Search**
    * Exhaustive search approach on fixed grid values
    * Suited for smaller search spaces
    * Quickly fail with growing size of search
2. **Random Search**
    * Suited for smaller search spaces
    * Quickly fail with growing size of search
3. **Bayesian Optimisation**
    * Assumes that a *specific probability distribution* is underlying the performance
    * Tested architectures constrain the probability distribution and guide the selection of the next option
    * In this way, promising architectures can be stochastically determined and tested 
4. **Evolutionary algorithms** 
    * ![evolutionary-search](assets/evolutionary-search.png)
5. **Reinforcement Learning** 
    * Agents goal is to maximize a reward
    * The available options are selected from the search space
    * The performance estimation strategy determines the reward
    * ![reinforcement-learning](assets/reinforcement-learning.png)


5.5 **Reinforcement Learning for NAS**
* ![reinforcement-learning-nas](assets/reinforcement-learning-nas.png)
## Measuring AutoML efficacy 
### Performance Estimation Strategy ![performance-estimation-strategy-automl](assets/performance-estimation-strategy-automl.png)

#### Strategy to reduce the cost
* **Lower fidelity estimates**
    * Reduce training time
        * *Data subset*
        * *Low resolution images*        
        * *Fewer filters and cells*    
    * Reduce cost but understimates performance
    * Works if relative ranking of architectures does not change due to lower fidelity estimates
    * Recent research show this is not the case
* **Learning curves extrapolation**
    * Requires predicting the learning curbe reliably 
    * Extrapolates bnased on initial learning
    * Removes poor performers
    * ![learning-curve-extrapolation](assets/learning-curve-extrapolation.png)
* **Weight inheritance/Network Morphisms**
    * Initialize weights of new architectures based on perviously trained architectures
        * Similar to transfer learning
    * Uses **Network Morphism** 
    * Underlying function unchanged 
        * New network inherits knowledge from parent network
        * Computationoal speed up: only a few days of GPU usage 
        * Network size not inherently bounded


## AutoML on cloud 
* **Amazon SageMaker Autopilot**
    * ![aws-sagemaker](assets/aws-sagemaker.png)
    * Key features:
        * ![aws-sagemaker-key-features](assets/aws-sagemaker-key-features.png)
    * Typical use cases:
        * ![aws-sagemaker-usecase](assets/aws-sagemaker-usecase.png)
* **Azure Automated ML** 
    * ![azure-automl](assets/azure-automl.png)
    * Key features
        * ![azure-key-features](assets/azure-key-features.png)
* **Google Cloud AutoML** 
    * ![gcp-automl](assets/gcp-automl.png)

![gcp-products-1](assets/gcp-products-1.png)
![gcp-products-2-edge](assets/gcp-products-2-edge.png)

