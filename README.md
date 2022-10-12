# Machine Learning Query System via Django
- This is an image classification system for HW1 of Security and Privacy of Machine Learning

# Installing
### Clone the repository
```=bash
    git clone https://github.com/shiannn/SPML-query-system.git
    cd SPML-query-system
```
### Create virtualenv
```=bash
    virtualenv -p <Path to Your Python> env
    source env/bin/activate
```
### Install dependencies
```=bash
    python -m pip install -r requirements.txt
```
### Apply
```=bash
    python source/manage.py migrate
```
### Collect static files (only on a production server)
```=bash
    python source/manage.py collectstatic
```
# Run
```=bash
    python source/manage.py runserver
```

# Acknowledgement
The codebase of the repo is highly borrowed from Yehor Smoliakov
- https://github.com/egorsmkv
- https://github.com/egorsmkv/simple-django-login-and-register