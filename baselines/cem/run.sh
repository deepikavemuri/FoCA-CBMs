cd cem
python setup.py install
cd ..

python experiments/run_experiments.py -c experiements/configs/awa2.yaml

sleep 30

python experiments/run_experiments.py -c experiements/configs/cifar100.yaml

sleep 30

python experiments/run_experiments.py -c experiements/configs/inet100.yaml