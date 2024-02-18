# Momentum_health

* **Step 1:** Clone repo and enter directory
    ```
    git clone https://github.com/cvdfoundation/kinetics-dataset.git
    cd kinetics-dataset
    ```

* **Step 2:** Creating directories
Please insert the following code line in your terminal.
    '''
    mkdir data plots save_models
    cd src
    '''
Please change the current_dir in [my_dir.yaml](src/configs/dirs/my_dir.yaml) to your current directory.

* **Step 3:** Training
    '''
    python train.py
    '''

* **Step 4:** Testing
    '''
    python test.py
    '''

